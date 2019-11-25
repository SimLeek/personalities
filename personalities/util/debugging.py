import torch
import gc
from collections import Counter


def get_tensors_in_memory():
    tensors_in_memory = []
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (
                hasattr(obj, "data") and torch.is_tensor(obj.data)
            ):
                tensors_in_memory.append((type(obj), obj.size()))
        except:
            pass
    return tensors_in_memory


def csv_log_tensors_in_memory(csv_filename="mem_dbg.csv"):
    with open(csv_filename, "a+") as mem_hist_file:
        for tens in get_tensors_in_memory():
            mem_hist_file.write(
                f"{str(tens[0]).replace(',', ';')}, {str(tens[1]).replace(',', ';')}, "
            )
        mem_hist_file.write("\n")


diff_mem = []


def csv_diff_log_tensors_in_memory(csv_filename="mem_diff_dbg.csv"):
    global diff_mem
    with open(csv_filename, "a+") as mem_hist_file:
        current_mem = get_tensors_in_memory()
        for tens in list(set(current_mem) - set(diff_mem)):
            mem_hist_file.write(
                f"{str(tens[0]).replace(',', ';')}, {str(tens[1]).replace(',', ';')}, "
            )
        diff_mem = current_mem
        mem_hist_file.write("\n")


def csv_dup_log_tensors_in_memory(csv_filename="mem_dup_dbg.csv"):
    with open(csv_filename, "a+") as mem_hist_file:
        mem_tensors = get_tensors_in_memory()
        dups = [
            (item[0], item[1], count)
            for item, count in Counter(mem_tensors).items()
            if count > 1
        ]
        for tens in dups:
            mem_hist_file.write(
                f"{str(tens[2])}, {str(tens[0]).replace(',', ';')}, {str(tens[1]).replace(',', ';')}, "
            )
        mem_hist_file.write("\n")
