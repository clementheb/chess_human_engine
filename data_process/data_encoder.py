import chess, chess.pgn
import numpy as np
import torch
import zstandard as zstd
import io
import csv
import os  
import data_process.config as cfg
import pandas as pd


def sq_to_2D(sq:int):
        row = 7 - (sq//8)
        col = sq % 8
        return(row,col)


def encode_move(move):     ###Attention ca ignore les sous-promotions, mais jpense on peut s'en blc pour l'instant 
        from_sq = move.from_square
        to_sq = move.to_square
        return(64*from_sq +to_sq)


def coord_to_algebraic(pos):
    """Convertit une position (row, col) en notation échecs (ex: 'e2')."""
    row, col = pos
    file = chr(ord('a') + int(col))
    rank = str(8 - int(row))   # row=0 -> rangée 8, row=7 -> rangée 1
    return f"{file}{rank}"


def decode_move_to_str(dec_move):
    """
    Prend le résultat de decode_4096to8x8 (un tuple (from_pos, to_pos), 
    ou une liste de tels tuples) et retourne la notation échecs (ex: 'e2e4').
    """
    if isinstance(dec_move, list):
        return [decode_move_to_str(m) for m in dec_move]
    
    from_pos, to_pos = dec_move
    return coord_to_algebraic(from_pos) + coord_to_algebraic(to_pos)


def decode_4096(enc_move, str_out = False):
    """
    Takes input integer between 0 and 4095 representing a move
    Outputs 2 tuples with column and line of starting move position and arriving position.
    """
    if isinstance(enc_move, (list,np.ndarray)):
        dec_moves = []
        for move in enc_move:
            dec_moves.append(decode_4096(move, str_out=str_out))  # propagé
        return(dec_moves)
    
    from_sq = enc_move // 64
    to_sq = enc_move % 64
    from_pos = np.array(sq_to_2D(from_sq))
    to_pos = np.array(sq_to_2D(to_sq))
    dec_move = (from_pos, to_pos)

    if str_out:
        result = decode_move_to_str(dec_move)
    else:
        result = dec_move
    return(result)


def sq_from_2d(row: int, col: int) -> int:
    return (7 - row) * 8 + col


def decode_board(x):
    """
    Rebuild a python-chess Board from your encoded tensor.

    x: torch.Tensor or np.ndarray of shape [8, 8, planes]

    """
    PIECE_TYPES = [
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
    chess.KING,
]

    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()

    board = chess.Board(None)

    # Pieces
    for row in range(8):
        for col in range(8):
            sq = sq_from_2d(row, col)

            # White pieces, planes 0-5
            for i, piece_type in enumerate(PIECE_TYPES):
                if x[i, row, col] > 0.5:
                    board.set_piece_at(
                        sq,
                        chess.Piece(piece_type, chess.WHITE)
                    )

            # Black pieces, planes 6-11
            for i, piece_type in enumerate(PIECE_TYPES):
                if x[i + 6,row, col] > 0.5:
                    board.set_piece_at(
                        sq,
                        chess.Piece(piece_type, chess.BLACK)
                    )
    # Turn
    board.turn = chess.WHITE if x[12,:, :].mean() > 0.5 else chess.BLACK

    # Castling rights
    board.castling_rights = chess.BB_EMPTY

    if x[13, :, :].mean() > 0.5:
        board.castling_rights |= chess.BB_A1

    if x[14,:, :].mean() > 0.5:
        board.castling_rights |= chess.BB_H1

    if x[15,:, :].mean() > 0.5:
        board.castling_rights |= chess.BB_A8

    if x[16,:, :].mean() > 0.5:
        board.castling_rights |= chess.BB_H8

    # En passant
    ep_plane = x[17, :, :]
    ep_indices = np.argwhere(ep_plane > 0.5)

    if len(ep_indices) > 0:
        row, col = ep_indices[0]
        board.ep_square = sq_from_2d(row, col)
    else:
        board.ep_square = None

    return board


def legal_moves(board, indices=False, encoding=encode_move):
    """
    Takes a postision (python chess position, not encoded)
    And returns the legal moves in np.array(bool), encoded by the function specified (default 4096 encoding for now)"""
    legals = board.legal_moves
    if encoding == encode_move:
        final = np.zeros(4096)
        legal_list = []
        for move in legals:
            legal_list.append(encoding(move))
        legal_set = set(legal_list)
        if not indices:
            for idx in range(4096): #goes through all the possible moves and checks if it is in the legal moves list
                final[idx] = int(idx in legal_set) #move encoded by 0 is impossible because it encodes from square 0 to square 0 (pieces dont stay immobile ofc)
    return(final)

class BoardEncoding:

    def __init__(self, use_ep=True, use_castling=True,use_attacks=True, use_elo=True):
        self.use_ep = use_ep
        self.use_castling = use_castling
        self.use_attacks = use_attacks
        
        self.n_planes = 12 #pieces, fixed
        self.n_planes += 1 #turn

        if use_elo:
            self.n_planes += 1 #elo

        if self.use_castling:
            self.n_planes += 4 #to encode the castling rights
        
        if self.use_ep:
            self.n_planes += 1

        # if self.use_attacks:
        #     self.n_planes += 1 most likely
        

    def pieces_encoding(self, board):       #encodes pieces position
        pieces = board.piece_map()
        array = np.zeros([12,8,8], dtype=np.float32)   #dimension needed just for this feature

        for sq, piece in pieces.items():
            row, col = sq_to_2D(sq)
            color_offset = 0 if piece.color else 6
            pc = piece.piece_type -1
            array[pc + color_offset, row, col] = 1
        return array


    def turn_encoding(self, ply):
        array = np.zeros([1,8,8], dtype=np.float32) #same here and everywhere else
        if ply % 1 == 0:
            array[0,:,:] = 1 ###1 if white, 0 if black
        return array


    def castling_encoding(self, board):        #Castling rights encoding
        array = np.zeros([4,8,8], dtype=np.float32) 

        array[0,:,:] = float(board.has_queenside_castling_rights(chess.WHITE))
        array[1,:,:] = float(board.has_kingside_castling_rights(chess.WHITE)) 
        array[2,:,:] = float(board.has_queenside_castling_rights(chess.BLACK))
        array[3,:,:] = float(board.has_kingside_castling_rights(chess.BLACK))
        return array

    def ep_encoding(self,board):         #En passant encoding
        array = np.zeros([1,8,8], dtype=np.float32)
        en_passant_sq = board.ep_square
        if en_passant_sq is not None:
            row, col = sq_to_2D(en_passant_sq)
            array[0, row, col] = 1
        return array


    def attacks_encoding(self, board):
        attacks = board.attacks()
        #Attack map (for tactical "sharpness")
        # to be implemented hein comme on dit bonne chance
    
    def encode_elo(self, elo):
        harr = np.zeros([1,8,8])
        norm_elo = (elo - cfg.FEATURES_ENCODING["elo_norm_lowbound"])/cfg.FEATURES_ENCODING["elo_norm_highbound"]
        harr[0,:,:] = norm_elo
        return(harr)
    
    def encode_all(self, board,ply, elo):
         arr = np.concatenate([
             self.pieces_encoding(board), #0-11
             self.turn_encoding(ply), #12
             self.castling_encoding(board), #13-16
             self.ep_encoding(board), #17
             self.encode_elo(elo), #18
                ], axis=0)
         return(torch.from_numpy(arr).float())


def games_to_skip(meta_data):
    if len(meta_data)==0:
        return(0)
    
    return(meta_data["nb_games"].sum())


def game_filtering(parsed_game):
    """
    Takes a game and returns the game if valid for the constraints defined (in config), else None
    """
    chess_header = parsed_game.headers
    header = {}
    for key, value in chess_header.items():
        header[key] = value

    #not elegant coding but it's to make the later code more readable
    #Basically renaming the config into easier variables
    elo_band = cfg.GAME_FILTERS['elo_band']
    time_control = cfg.GAME_FILTERS['time_control']
    ok_elo_diff = cfg.GAME_FILTERS['ok_elo_diff']
    rated_only = cfg.GAME_FILTERS['rated_only']

    #elo band filter
    try:
        whiteelo = int(header["WhiteElo"])
        blackelo = int(header["BlackElo"])
    except ValueError:
        return None

    header["elo"] = 1/2*(whiteelo + blackelo)
    if (header["elo"] < elo_band[0]) or (header["elo"] > elo_band[1]):
        return None
    
    #time control filter
    if header["TimeControl"] != time_control:
        return None    
    #elo diff
    if np.abs(whiteelo - blackelo)>ok_elo_diff:
        return None

    #fitlers games rated only, by checking the Event (prevents non-competitive games in the dataset)
    if rated_only:
        if not "Rated" in header["Event"]:
            return None

    return(parsed_game, header)


def encode_pgn(stream, chunk_size, rebuild:bool = False, nb_encoded_games:int=0): #main function calling everything
    X = []
    Y = []
    Headers = []
    done = False
    encoder = BoardEncoding()
    samples = 0 ; nb_games=0

    while samples <= chunk_size and not done:
        if not rebuild:
            for skip_count in range(nb_encoded_games):
                chess.pgn.skip_game(stream)            #skips the already encoded games if rebuild is false (default)
                if skip_count % 200 == 0 :
                    print(f"{skip_count} games skipped, already in data")
            print(f"{nb_encoded_games} games skipped in total, already in data")
        game = chess.pgn.read_game(stream)
        if game is None:
            break      
        filtered = game_filtering(game)

        if filtered == None:
            continue
        else:
            header = filtered[1]
        Headers.append(header)
        elo = header['elo']

        board = game.board()
        ply = 1
        for move in game.mainline_moves():
            X.append(encoder.encode_all(board,ply,elo))
            Y.append(encode_move(move))

            board.push(move)
            samples += 1 ; ply += 1/2 #Weird ply that takes n,5 values when it's move n and black to play. supervises turn_encoder
        nb_games += 1
    if samples < chunk_size:
        done = True
    print(f"{samples} samples envoyes dans X et Y")
    return((X,Y), Headers, samples, nb_games)


def load_meta_data(path, rebuild):
    if os.path.exists(path) and not rebuild:
        df = pd.read_parquet(path=path)
        print(f"Meta data loaded from {path}")
        return(df)
    return(pd.DataFrame(columns= [
                                "chunk_id",
                                "nb_games",
                                "chunk_samples",
                                "save_name",
                                "elo_low",
                                "elo_high",
                                "time_control",
                                "ok_elo_diff",
                                "rated_only",
                                ]))


def build_data_final(data_path,
                     meta_data_path,
                     max_chunks,chunk_size, 
                     save_folder_path, 
                     meta_data_save_path):
    done = False
    meta_data = load_meta_data(meta_data_path,cfg.REBUILD_DATA)
    nb_enc_games = games_to_skip(meta_data=meta_data)
    if not cfg.REBUILD_DATA:
        iter = len(meta_data)
    else:
        iter = 0

    with open(data_path, "rb") as f:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(f) as reader:
            text_stream = io.TextIOWrapper(reader, encoding="utf-8")
            if iter>max_chunks:
                print(f"Max chunk number already met ({iter}/{max_chunks}), increase max_chunks if you want to build more data")
            while not done and iter < max_chunks :
                (X, Y), header, chunk_samples, nb_games = encode_pgn(text_stream,chunk_size, rebuild=cfg.REBUILD_DATA, nb_encoded_games=nb_enc_games)

                elo_low_bond, elo_high_bound = cfg.GAME_FILTERS['elo_band']
                save_name = save_folder_path+f"chunk_{iter}_{elo_low_bond}_{elo_high_bound}.npz"

                X_np = torch.stack(X).numpy().astype(np.float32)
                Y_np = np.array(Y, dtype =np.int64)
                np.savez(save_name, X=X_np, Y=Y_np)   #to pass meta feautres to model, encode them with the position using the class BoardEncoding
                print(f"chunk {iter} saved in {save_folder_path}chunk_{iter}.npz")

                #Meta data on chunk level : number of games, sambles in the chunk, and the filters applied on the dataset 
                meta_data.loc[len(meta_data)] = ({

                    "chunk_id":iter,
                    "nb_games":nb_games,
                    "chunk_samples":chunk_samples,
                    "save_name":save_name,

                    "elo_low":cfg.GAME_FILTERS["elo_band"][0],
                    "elo_high":cfg.GAME_FILTERS["elo_band"][1],

                    "time_control":cfg.GAME_FILTERS["time_control"],
                    "ok_elo_diff":cfg.GAME_FILTERS["ok_elo_diff"],
                    "rated_only":cfg.GAME_FILTERS["rated_only"]
                })

                # print(f"meta_data:{meta_data}")
                iter+=1
                done = chunk_samples<chunk_size  ##if chunk samples are below chunk size it means the data has run out, i.e no games to parse left in the pgn. Not likely to happen.

            meta_data.to_parquet(meta_data_save_path, index=False)
            print(f"meta_data saved in {meta_data_save_path}")


if __name__ == "__main__":
    build_data_final(data_path="data/lichess_db_standard_rated_2026-02.pgn.zst",
                     meta_data_path="data/meta_data_npz.pqt",
                     max_chunks=cfg.MAX_CHUNKS,
                     chunk_size=cfg.CHUNK_SIZE,
                     save_folder_path="data/npz/",
                     meta_data_save_path="data/meta_data_npz.pqt")



    ###SAVING IN CASE EVERYTHING IS BROKEN
    # done = False
    # meta_data = load_meta_data("data/meta_data_npz.csv")
    # nb_enc_games = games_to_skip(meta_data=meta_data)
    # if not REBUILD_DATA:
    #     iter = len(meta_data)
    # else:
    #     iter = 0

    # with open("data/lichess_db_standard_rated_2026-02.pgn.zst", "rb") as f:
    #     dctx = zstd.ZstdDecompressor()
    #     with dctx.stream_reader(f) as reader:
    #         text_stream = io.TextIOWrapper(reader, encoding="utf-8")
    #         if iter>MAX_CHUNKS:
    #             print(f"Max chunk number already met ({iter}/{MAX_CHUNKS}), increase MAX_CHUNKS if you want to build more data")
    #         while not done and iter < MAX_CHUNKS :
    #             (X, Y), chunk_samples, nb_games = encode_pgn(text_stream,CHUNK_SIZE, rebuild=True, nb_encoded_games=nb_enc_games)
    #             np.savez(f"data/npz/chunk_{iter}_fullDS.npz", X=np.array(X), Y=np.array(Y))

    #             print(f"chunk {iter} sauvegardé à data/npz/chunk_{iter}_fullDS.npz")
    #             meta_data[iter] = [nb_games,chunk_samples]
    #             print(f"meta_data:{meta_data}")

    #             iter+=1
    #             done = (chunk_samples<CHUNK_SIZE)  ##if chunk samples are below chunk size it means the data has run out.

    #         with open('data/meta_data_npz.csv', 'w', newline='') as output:     #saving metadata for future use/cacheing
    #             writer = csv.writer(output)
    #             for key, value in meta_data.items():
    #                 writer.writerow([key, value])
    #         print("meta_data saved")