import sys
import os
import csv
import statistics

def parse_benchmark_name(bench):
    params = bench.split("/");
    assert(len(params) == 2);
    num_cons = params[0].split("-")[-1];
    op = params[1];
    if op == "ProofSize": 
        op = "Proof Size (B)";
    if op == "Prove":
        op = "Prove (ms)"
    if op == "Verify":
        op = "Verify (ms)";
    num_cons_ver_circuit = 20764;
    return [int(num_cons) + num_cons_ver_circuit, op]
                    
def parse(fname):
    res = [];
    with open(fname, 'r') as f:
        while True:
            line = f.readline();
            if not line:
                break;
            if "ProofSize" in line:
                entries = line.split(":");
                if len(entries) != 2:
                    print("ERROR: Wrong format of line" + line);
                    sys.exit()
                size = entries[1][:-1].strip().split(" ")[0];
                res1 = parse_benchmark_name(entries[0]);
                res1.append(size);
                res1.append(size);
                res1.append(size);
                res.append(res1);
            else:
                entries = list(filter(lambda x: x != "", line.split(" ")));
                if "Benchmarking" in entries:
                    idx = entries.index("Benchmarking");
                    cur_benchmark = entries[idx + 1][:-1];
                if "time:" in entries:
                    idx = entries.index("time:");
                    time = entries[idx + 3]; 
                    time_units = entries[idx + 4];
                    min_time = entries[idx + 1][1:]; 
                    assert(time_units == entries[idx + 2]);
                    max_time = entries[idx + 5];
                    assert(time_units == entries[idx + 6][:-2]);
                    if time_units == "s": 
                        time = float(time)*1000;
                        min_time = float(min_time)*1000;
                        max_time = float(max_time)*1000;
                    else:
                        assert(time_units == "ms");
                    if (cur_benchmark == ""):
                        print("ERROR: Problem while parsing the file, found time without benchmark name");
                        sys.exit();
                    res1 = parse_benchmark_name(cur_benchmark);
                    res1.append(time);
                    res1.append(min_time);
                    res1.append(max_time);
                    res.append(res1);
    return res;

def write_results_to_csv(results, csv_fname):
    fieldnames = ["Num Constraints", "Operation", "Center value", "Confidence interval min", "Confidence interval max"]
    with open(csv_fname, mode = "w") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames);
        writer.writeheader();
        while len(results) > 0:
            [num_cons, op, time, min_time, max_time] =  results.pop();
            writer.writerow({
                "Num Constraints": num_cons,
                "Operation": op,
                "Center value": time,
                "Confidence interval min": min_time,
                "Confidence interval max": max_time,
            });

if __name__ == "__main__":
    results = parse("compressed-snark.txt");
    write_results_to_csv(results, "compressed-snark.csv")
