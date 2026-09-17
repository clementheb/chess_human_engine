##Processed data parameters
MAX_CHUNKS = 20   #number of chunks max done
CHUNK_SIZE = 5000 #Careful, this will go until the 5000th move, and finish the game being analysed before stopping, leading to real chunk sizes of 5000+end of game (generally less than 100 moves)
REBUILD_DATA = False   #If True it rebuilds all the data instead of directly going to the last one

CHUNKS_LOADED = 5 #None to load all chunks

PGN_PATH = "data/lichess_db_standard_rated_2026-02.pgn.zst"
DATA_FOLDER = "data/npz/"
META_PATH = "data/meta_data_npz.pqt"


DATA_FILTERING_LIMIT = 10_000 #in number of games


META_FEATURES = [
    "Site",
    "WhiteElo",
    "BlackElo",
    "Opening",
    "TimeControl",
    "Result",
    "Event"
]

time_controls = {}  ##dict of time controls, just to have eg. time_controls["rapid"] instead of the ugly '600+0' but for the moment flemme de ouf

GAME_FILTERS = {
    "elo_band":[1600,1800],
    "ok_elo_diff":100,
    "time_control":"600+0",
    "rated_only":True
}

FEATURES_ENCODING = {
    "elo_norm_lowbound":600,
    "elo_norm_highbound":3000,
}