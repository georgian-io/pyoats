





class Scorer():
    def __init__(self):
        self.stats = {}

    def add_one_result(self, ground_truth, model_results, model_names):
        stats = {}
        for idx, m in enumerate(model_results):
            tp, tn, fp, fn = 0, 0, 0, 0
            model_name = model_names[idx]

            train_len, test_len = ground_truth[f_name]["train_len"], ground_truth[f_name]["test_len"]
            anom_start, anom_end = ground_truth[f_name]["anom_start"], ground_truth[f_name]["anom_end"]

            arr = np.loadtxt(f)
            th = get_anomaly_threshold(1e-3, arr)

            is_anom = arr.copy()

            anom_indices = [i + train_len for i in np.where(arr > th)[0]]

            anom_len = anom_end-anom_start+1
            match = sum(anom_indices.count(i) for i in range(anom_start-100, anom_end+100+1))
            if match > 0: tp += anom_len
            else: fn += anom_len
            fp += len(anom_indices) - match
            tn += test_len - tp - fn - fp

            try: stats[model_name]
            except KeyError: stats[model_name] = {}

            try: stats[model_name]["tp"] += tp
            except KeyError: stats[model_name]["tp"] = tp
            try: stats[model_name]["tn"] += tn
            except KeyError: stats[model_name]["tn"] = tn
            try: stats[model_name]["fp"] += fp
            except KeyError: stats[model_name]["fp"] = fp
            try: stats[model_name]["fn"] += fn
            except KeyError: stats[model_name]["fn"] = fn
