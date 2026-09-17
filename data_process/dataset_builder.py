import chess, chess.pgn
import numpy as np
import zstandard as zstd
import io
import pandas as pd
from torch.utils.data import Dataset
import data_process.config as cfg
from sklearn.model_selection import train_test_split

###Dataset class for PyTorch

class LichessGames(Dataset):
    def __init__(self, frac=1.0, seed=42):
        """
        frac: fraction of rows to keep from each loaded chunk (0 < frac <= 1)
        """
        self.mode = 'train'
        self.chunks = cfg.CHUNKS_LOADED
        rng = np.random.default_rng(seed)

        meta_data = pd.read_parquet(cfg.META_PATH)
        chunk_ids = list(meta_data['chunk_id'])

        if self.chunks is not None:
            chunk_ids = list(rng.choice(chunk_ids, size=min(self.chunks, len(chunk_ids)), replace=False))

        X, Y = [], []
        for chunk_id in chunk_ids:
            path = str(meta_data.loc[chunk_id]['save_name'])
            # mmap so we don't pull the whole chunk into RAM before subsampling
            chunk = np.load(path, mmap_mode='r')
            n = chunk['X'].shape[0]

            if frac < 1.0:
                keep = max(1, int(n * frac))
                idx = rng.choice(n, size=keep, replace=False)
                idx.sort()  # mmap slicing is friendlier with sorted indices
                X.append(np.asarray(chunk['X'][idx], dtype=np.float32))
                Y.append(np.asarray(chunk['Y'][idx], dtype=np.int64))
            else:
                X.append(np.asarray(chunk['X'], dtype=np.float32))
                Y.append(np.asarray(chunk['Y'], dtype=np.int64))

        self.input = np.concatenate(X, axis=0)
        self.moves = np.concatenate(Y, axis=0)

        self.x_train = self.x_test = self.y_train = self.y_test = np.zeros(1, dtype=np.float32)

    def __getitem__(self, index):
        if self.mode == 'train':
            sample = {"position":self.x_train[index], "move": self.y_train[index]}
        elif self.mode == 'test':
            sample = {"position":self.x_test[index], "move": self.y_test[index]}
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        return(sample)


    def __len__(self):
        if self.mode == 'train':
            return(self.x_train.shape[0])
        if self.mode == 'test':
            return(self.x_test.shape[0])
        
    def train_test(self, train_size, test_size):
        self.x_train, self.x_test, self.y_train, self.y_test = \
        train_test_split(self.input, self.moves, train_size=train_size, test_size=test_size, random_state=42)

    # def sample(self, number:int):
    #     self.input = 


###The following code is building a meta-data dataset from headers only, to explore those features etc.
def features_dataset_creator(cfg): ##ptet plus polyvalent si ca prend en entrée un pgn au lieu d'un path; a voir.
    games_checked = 0
    dataset = {}
    # dataset['headers'] = cfg.META_FEATURES

    with open(cfg.PGN_PATH, "rb") as f:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(f) as reader:
            text_stream = io.TextIOWrapper(reader, encoding="utf-8")    
            while games_checked < cfg.DATA_FILTERING_LIMIT:
                game_features = chess.pgn.read_headers(text_stream)
                if game_features is None:
                        break
                games_checked += 1
                game_site = game_features["Site"]
                dataset[game_site] = []
                for feature in cfg.META_FEATURES[1:]:
                        dataset[game_site].append(game_features[feature])
    df = pd.DataFrame.from_dict(dataset, orient="index", columns=cfg.META_FEATURES[1:])
    for feature in ["WhiteElo","BlackElo"]:
        df[feature] = pd.to_numeric(df[feature])
    return df


def features_dataset_filtering(cfg, 
                      elo_band,
                      save_path=None,
                      time_control=None,
                      ok_elo_diff=None,
                      rated_only:bool=True,
                      save_in_csv=False):
    """
    -ok_elo_diff is to filter out games with too high of an elo difference between the two players,
    because not representative of the elo bracket. Should not filter too much since games online below 2200 
    have low elo difference anyway. Only at high elo it becomes high because players are scarcer.
    -elo_band is a 2-elements list with first lower, and second higher elo bound. Will be taken from the average elo between the two players
    -save_path = None will not save the file
    """

    df = features_dataset_creator(cfg)

    if time_control is not None:
        df = df[df["TimeControl"]==time_control]
    
    if ok_elo_diff is not None:
        df['diff_elo'] = (df["WhiteElo"] -  df["BlackElo"]).abs()
        df = df[df["diff_elo"] < ok_elo_diff]

    df['avg_elo'] = 1/2*(df["WhiteElo"]+ df["BlackElo"])
    df = df[df["avg_elo"]>elo_band[0]]
    df = df[df["avg_elo"]< elo_band[1]]

    df["norm_elo"] =  (df['avg_elo'] - cfg.FEATURES_ENCODING["elo_norm_lowbound"])/cfg.FEATURES_ENCODING["elo_norm_highbound"]
    print(df['norm_elo'].describe())

    if rated_only:
        df = df[df["Event"].str.contains("Rated")]
    
    if save_path is not None:
        if save_in_csv:
            df.to_csv(save_path, encoding='utf-8')
        else:
             df.to_parquet(save_path)
    return df


def exploration():
    df = pd.read_csv("data/df_filtered.csv", encoding='utf-8')
    df_un = features_dataset_creator(cfg=cfg)
    df['diff_elo'] = (df["WhiteElo"] -  df["BlackElo"]).abs()
    df_un['diff_elo'] = (df_un["WhiteElo"] -  df_un["BlackElo"]).abs()
    df["norm_elo"] =  (df['avg_elo'] - cfg.FEATURES_ENCODING["elo_norm_lowbound"])/cfg.FEATURES_ENCODING["elo_norm_highbound"]
    print(df['norm_elo'].describe())


    # df = df[df["diff_elo"] < 100]

    for feature in ["WhiteElo","BlackElo", "diff_elo"]:
        print("Filtered dataset")
        print(df[feature].describe())
        print('')
        print("Unfiltered dataset")
        print(df_un[feature].describe())
        print('')





# if __name__ == "__main__":
    #  dataset_filtering(config,
    #                    [1600,1800],
    #                    save_path="data/df_filtered.csv",
    #                    ok_elo_diff=100,
    #                    time_control="600+0",
    #                    save_csv=True)

# exploration()