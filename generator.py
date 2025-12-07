import numpy as np
import pickle
import os


# =========================================================
# CONFIG — MUST MATCH DataHandler expectations
# =========================================================
ROW = 8           # = args.row
COL = 8           # = args.col
CATEGORIES = 4    # = args.offNum
TOTAL_DAYS = 2000

TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.1   # remaining 0.2 goes to test


# =========================================================
# 1. Spatial adjacency for grid
# =========================================================
def build_grid_adjacency(row, col):
    N = row * col
    A = np.zeros((N, N), dtype=np.float32)

    def idx(r, c): return r * col + c

    for r in range(row):
        for c in range(col):
            i = idx(r, c)
            # 4-neighbors grid
            for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
                rr, cc = r+dr, c+dc
                if 0 <= rr < row and 0 <= cc < col:
                    j = idx(rr,cc)
                    A[i,j] = 1
            A[i,i] = 1  # self loop

    # row-normalize
    deg = A.sum(axis=1, keepdims=True)
    deg[deg==0] = 1
    A = A/deg
    return A


# =========================================================
# 2. Generate synthetic crime data
# =========================================================
def generate_crime_tensor(days=2000, row=ROW, col=COL, cate=CATEGORIES):
    area = row * col
    t = np.arange(days)

    # global cycles
    weekly = 0.5 * np.sin(2*np.pi*t/7)
    monthly = 0.5 * np.sin(2*np.pi*t/30)

    # correlated latent factors
    latent_A = weekly + 0.3*monthly
    latent_B = 0.5*monthly + 0.5*np.sin(2*np.pi*t/14)

    data = np.zeros((row, col, days, cate), dtype=np.float32)

    # spatial adjacency for diffusion
    A = build_grid_adjacency(row, col)
    S = A  # diffusion

    for r in range(row):
        for c in range(col):
            idx = r*col + c

            region_bias = 0.3*np.sin(0.01*t*idx)

            for k in range(cate):
                base = 1 + 0.1*np.random.randn()

                if k in (0,1):  # correlated pair
                    mu = base + latent_A + 0.5*region_bias
                else:
                    mu = base + latent_B + 0.4*region_bias

                mu = np.clip(mu, 0.01, None)
                data[r,c,:,k] = mu

    # Diffusion across grid via adjacency matrix
    flat = data.reshape((row*col, days, cate))  # (area, T, C)

    for d in range(days):
        flat[:,d,:] = S @ flat[:,d,:]

    data = flat.reshape((row, col, days, cate))

    # Poisson sampling
    data = np.random.poisson(0.5 * data).astype(np.float32)

    # Add crime bursts
    for _ in range(100):
        rr = np.random.randint(0,row)
        cc = np.random.randint(0,col)
        day0 = np.random.randint(0,days)
        cat = np.random.randint(0,cate)
        burst = np.exp(-((t-day0)**2)/(2*2**2))
        burst *= np.random.uniform(3,6)
        data[rr,cc,:,cat] += burst.astype(np.float32)

    return data


# =========================================================
# 3. Save in STHSL format
# =========================================================
def save_dataset(folder):
    os.makedirs(folder, exist_ok=True)

    data = generate_crime_tensor(days=TOTAL_DAYS)

    # Split along time dimension
    trn_end = int(TOTAL_DAYS * TRAIN_SPLIT)
    val_end = int(TOTAL_DAYS * (TRAIN_SPLIT + VAL_SPLIT))

    trn = data[:,:, :trn_end, :]
    val = data[:,:, trn_end:val_end, :]
    tst = data[:,:, val_end:, :]

    # MUST SAVE RAW NDARRAY ONLY (NO DICT)
    with open(f"{folder}/trn.pkl", "wb") as f: pickle.dump(trn, f)
    with open(f"{folder}/val.pkl", "wb") as f: pickle.dump(val, f)
    with open(f"{folder}/tst.pkl", "wb") as f: pickle.dump(tst, f)

    # Build hypergraph H (spatial communities)
    area = ROW*COL
    H = np.zeros((area, 6), dtype=np.float32)
    coords = np.array([[r,c] for r in range(ROW) for c in range(COL)])

    for e in range(6):
        center = coords[np.random.randint(0,area)]
        d = np.linalg.norm(coords - center, axis=1)
        members = np.argsort(d)[:10]  # nearest 10 nodes
        H[members, e] = 1

    np.save(f"{folder}/H.npy", H)

    print("✔ Synthetic dataset saved.")
    print("trn:", trn.shape)
    print("val:", val.shape)
    print("tst:", tst.shape)
    print("H:", H.shape)


# Run generator
save_dataset("Datasets/NYC_SYN")

