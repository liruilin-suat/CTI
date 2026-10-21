import pandas as pd
import os



# prepare case study test
compound_ids = {"TCMsuite": [215, 3586], "Symmap": [43, 630]}

datasets = ["TCMsuite", "Symmap"]

save_dir = "datasets/{}/case_study"

target_ids_path = "datasets/{}/target_ids.txt"

herb_compound_path = "datasets/{}/herb_compound_cleaned.txt"

for d in datasets:
    target_ids = pd.read_csv(target_ids_path.format(d), header=None)

    d_compound_ids = compound_ids[d]

    h_c_df = pd.read_csv(herb_compound_path.format(d), sep="\t", header=None)

    h_c_df_clean = h_c_df[~h_c_df[1].isin(d_compound_ids)]
    h_c_df_clean.to_csv(os.path.join(save_dir.format(d), "herb_compound_cleaned.txt"), sep="\t", header=None, index=False)

    test_set = []
    for c_id in d_compound_ids:
        test_set.extend([[c_id, t_id[0], -1] for t_id in target_ids.values.tolist()])
    
    pd.DataFrame(test_set).to_csv(os.path.join(save_dir.format(d), "test_all.txt"), index=False, sep="\t", header=None)



