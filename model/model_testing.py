import torch
from data_process.data_encoder import sq_from_2d
from data_process.dataset_builder import LichessGames
from torch.utils.data import DataLoader
from model.model_first import CNN_base
import model.model_cfg as cfg
from data_process.data_encoder import decode_board, legal_moves, decode_4096, decode_move_to_str
import numpy as np



MODEL_LOAD_PATH = "training/baseline_model/baseline_model_50E_1618"
MOVES_OUTPUT = 5


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = CNN_base().to(device)

def game_output_analysis(model, idx):

    data = LichessGames()    
    data.train_test(0.8,0.2)
    data.mode = 'test'
    test_data = DataLoader(data, batch_size=16, shuffle=True)

    model.load_state_dict(torch.load(MODEL_LOAD_PATH, weights_only=True, map_location=device))
    model.eval()

    sample_idx = idx  # ou random.randint(0, len(data)-1) pour un sample aléatoire
    sample = data[sample_idx]  # dict {"position": ..., "move": ...}

    position = sample['position']
    move = sample['move']

    torch_position = torch.as_tensor(position, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        model_out = model(torch_position)
        logits = torch.softmax(model_out, dim = 1)

    logits = logits.squeeze(dim=0) #enlève le batch dim
    
    np_logits = logits.cpu().detach().numpy() 
    board = decode_board(position)
    mask = legal_moves(board)
    turn = 'WHITE' if board.turn else 'BLACK'


    topk_all = np_logits.argsort()[::-1][:MOVES_OUTPUT]

    legal_logits = mask*np_logits #legal output
    topk_legals = legal_logits.argsort()[::-1][:MOVES_OUTPUT]
     
    top_legal_logits = np.sort(legal_logits)[::-1][:MOVES_OUTPUT] 
    top_logits = np.sort(np_logits)[::-1][:MOVES_OUTPUT]

    out_moves = decode_4096(topk_all, str_out=True)
    out_legal = decode_4096(topk_legals, str_out = True)

    print(board)
    print(turn, 'to play')
    print("Game FEN: ")
    print(board.fen(en_passant = 'fen'))

    print(f'Top {MOVES_OUTPUT} Legal moves output:')
    print(out_legal)
    print(top_legal_logits, "associated logits")
    print(f'Top {MOVES_OUTPUT} all moves output (no legal filter applied):')
    print(out_moves)
    print(top_logits, "associated logits")
    print("Played move in the game: ")
    print(decode_move_to_str(decode_4096(move)))


for idx in np.random.randint(0,200,3):
    print('NEW GAME')
    game_output_analysis(model=model, idx=idx)
    print('')

