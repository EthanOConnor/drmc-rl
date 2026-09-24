"""Sample covered placements from every monthly release into one small parquet (mombox)."""
import glob, sys
import numpy as np
import pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc

out_root, per_month, dest = sys.argv[1], int(sys.argv[2]), sys.argv[3]
cols = ['decision_id', 'player', 'day', 'speed', 'speed_ups', 'tau_frames', 'input_frames', 'held_before_spawn',
        'held_at_spawn', 'input_rle_u16_u8', 'lock_x', 'lock_y_top', 'lock_rotation', 'lock_repaired', 'field',
        'horizontal_velocity', 'speed_counter', 'frame_counter', 'pill_left', 'pill_right']
rng = np.random.default_rng(20260924)
tables = []
releases = sorted(glob.glob(f'{out_root}/releases/timing-*'))
for rel in releases:
    for f in glob.glob(f'{rel}/decisions/*/*/*.parquet'):
        t = pq.read_table(f, columns=cols)
        t = t.filter(pc.and_(pc.greater(t['input_frames'], 0), pc.invert(pc.is_null(t['held_before_spawn']))))
        if t.num_rows > per_month:
            t = t.take(np.sort(rng.choice(t.num_rows, per_month, replace=False)))
        tables.append(t)
        print(f, t.num_rows, flush=True)
ratings = pq.read_table(glob.glob(f'{releases[-1]}/ratings/*.parquet')[0])
pq.write_table(pa.concat_tables(tables), dest + '.decisions.parquet')
pq.write_table(ratings, dest + '.ratings.parquet')
print('releases', len(releases))
