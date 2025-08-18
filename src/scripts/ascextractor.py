import os
import json
import pandas as pd

class AscExtractor:
    def __init__(self):
        self.column_names = ["start_ms", "end_ms", "length_ms", "x", "y"]

    def extract(
            self, asc_data_folder: str, 
            output_folder_base: str, 
            save_output: bool = True, 
            do_only_for_subjects: list[str] = [], 
            skip_practice_trials = ["-3", "-2", "-1"]
        ):

        """ Params:
        - asc_data_folder: where the  .asc files are located
        - output_folder_base: where the output file structure will be created
        - save_output: save the resulting folder structure, or just print the processing result
        - do_only_for_subjects: list of subject numbers to extract, others are skipped
        """

        for file in os.listdir(asc_data_folder):

            name, ext = file.split(".")
            if ext == "asc":

                file_path = os.path.join(asc_data_folder, file)
                _, subject = name.split("_")
            
                if do_only_for_subjects and (subject not in do_only_for_subjects):
                    continue

                with open(file_path, "r") as f:
                    lines = f.readlines()

                # move to the "MSG start_trial line"
                current_line = 0
                while True:
                    if "start_trial" in lines[current_line]:
                        break
                    else:
                        current_line += 1
                start_line = current_line + 1

                # move to the first trial
                while True:
                    if lines[start_line][0:3] == "MSG":
                        if "trial-number" in lines[start_line]:
                            break
                    else:
                        start_line += 1

                print(f"start_line for {subject=}:", start_line, lines[start_line].strip())

                left_data: dict[str, list[tuple]] = {}
                right_data: dict[str, list[tuple]] = {}
                images: dict[str, str] = {}

                current_trial = 0

                for line in lines[start_line:]:

                    # process message
                    if line[0:3] == "MSG":
                        if ("_end;" in line) and ("recalibration_end;" not in line):
                            print(f"end_line for {subject=}", line.strip())
                            break # fx task_1_end, there are two tasks in each

                        if "trial-number" in line:
                            msg, time_ms, info = line.split("\t")
                            trigger_number, trigger_name, trial_number, stimuli = info.split(";")
                            _, n = trial_number.strip().split(" ")
                            current_trial = n
                            if ((current_trial not in left_data) and (current_trial not in right_data)):
                                left_data[current_trial] = []
                                right_data[current_trial] = [] 
                            _, img = stimuli.strip().split(" ")
                            images[current_trial] = img

                    # process fixation
                    elif line[0:4] == "EFIX":
                        msg, eye, start_ms, end_ms, length_ms, x, y, pupil_size = line.split("\t")
                        if eye == "L":
                            left_data[current_trial].append((start_ms, end_ms, length_ms, x, y))
                        elif eye == "R":
                            right_data[current_trial].append((start_ms, end_ms, length_ms, x, y))
                
                # remove practice trials -3, -2, -1 
                for trial_to_skip in skip_practice_trials:
                    if trial_to_skip in left_data:
                        del left_data[trial_to_skip]
                    if trial_to_skip in right_data:
                        del right_data[trial_to_skip]
                    if trial_to_skip in images:
                        del images[trial_to_skip]

                if save_output:

                    subject_folder = f"{output_folder_base}/sub_{subject}"
                    os.makedirs(subject_folder, exist_ok=True)

                    for k, v in left_data.items():
                        df = pd.DataFrame(data=v, columns=self.column_names)
                        df.to_csv(f"{subject_folder}/trial_{k}.csv", sep="\t", index=False)
                    
                    with open(f"{subject_folder}/images.json", "w") as f:
                        json.dump(images, f, indent=4)

                print(f"{subject=} processed")
                print(len(left_data), "trials processed")
                print("")


if __name__ == "__main__":
    
    extractor = AscExtractor()
    extractor.extract(asc_data_folder="eyelink_raw/asc/", output_folder_base="eyelink_data/scanpaths/", save_output=True)