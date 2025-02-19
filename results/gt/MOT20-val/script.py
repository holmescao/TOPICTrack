
import pdb

files = {
    "MOT20-01/gt/gt.txt": (216, 429),
    "MOT20-02/gt/gt.txt": (1393, 2782),
    "MOT20-03/gt/gt.txt": (1204, 2405),
    "MOT20-05/gt/gt.txt": (1659, 3315),
}


for f_name, val_range in files.items():
    new_data = []
    with open(f_name, "r") as fp:
        for line in fp:
            tokens = line.split(",")
            if int(tokens[0]) < val_range[0]:
                continue
            tokens[0] = str(int(tokens[0]) - val_range[0] + 1)
            new_data.append(",".join(tokens))
    with open(f_name, "w") as fp:
        fp.writelines(new_data)
