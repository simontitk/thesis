from PIL import Image
import pandas as pd
import os
import json
from typedefs import AOICenters
"""
REMARKS:

in the answers, the first row for each subject is just NA values

for each subject, EXCEPT for sub_1, there are 3 images which are just safe practices
Images used for -3, -2, -1:
- 2017_12666203.jpg
- 2017_13226304.jpg
- 2017_13479904.jpg

"""

class DataLoader:
    """
    image_map: dict(image name, dict(subject, trial)        
    """

    def __init__(self, root_folder: str):
        self.root_folder = root_folder
        self.answers_folder = os.path.join(root_folder, "answers")
        self.images_folder = os.path.join(root_folder, "images")
        self.scanpaths_folder = os.path.join(root_folder, "scanpaths")
        self.image_map: dict[str, dict[str, ]] = {}
        self.answer_columns = ["response", "label"]
        self.image_width = 1250
        self.image_height = 740

        scanpath_subfolders = sorted(os.listdir(self.scanpaths_folder), key=lambda d: int(d.split("_")[1]))
        
        for dir in scanpath_subfolders:
            subject = int(dir.split("_")[1])
            images_json_path = os.path.join(self.scanpaths_folder, dir, "images.json")

            with open(images_json_path, "r") as f:
                images_json: dict = json.load(f)
                for trial, image in images_json.items():
                    if image in self.image_map:
                        self.image_map[image][subject] = int(trial)
                    else:
                        self.image_map[image] = {subject: int(trial)}

        aoi_df = pd.read_excel(os.path.join(root_folder, "aois.xlsx"))
        self.image_aois: dict[int | str, dict[int, tuple[int, int]]] = {}
        for img, group in aoi_df.groupby("img"):
            aoi_dict = {}
            for _, row in group.iterrows():
                aoi_dict[int(row["AOI"])] = (int(row["x"]), int(row["y"]))
            self.image_aois[img] = aoi_dict


    def get_subject_images(self, subject: int) -> dict[str, str]:
        with open(os.path.join(self.scanpaths_folder, f"sub_{subject}", "images.json"), "r") as f:
            images = json.load(f)
        return images
    
    
    def get_subject_image(self, subject: int, trial: int, file: bool = False) -> str:
        img = self.get_subject_images(subject)[str(trial)]
        if file:
            return self.get_image(img)
        return img


    def get_subject_trial(self, subject: int, trial: int) -> pd.DataFrame:
        path = os.path.join(self.scanpaths_folder, f"sub_{subject}", f"trial_{trial}.csv")
        df = pd.read_csv(path, sep="\t")
        df["x"] = self.image_width - df["x"] # invert the x axis to match the image
        df = df[(df["x"] >= 0) & (df["x"] <= self.image_width)].reset_index(drop=True) # remove the outliers from (0, 1250)
        df = df[(df["y"] >= 0) & (df["y"] <= self.image_height)].reset_index(drop=True) # remove the outliers from (0, 740)
        return df
    

    def get_subject_answers(self, subject: int) -> pd.DataFrame:
        path = os.path.join(self.answers_folder, f"subject-{subject}.csv")
        number_of_rows_to_skip = 1 if subject == 1 else 4
        df = pd.read_csv(path, usecols=self.answer_columns, skiprows=range(1, 1+number_of_rows_to_skip))
        
        # some responses have a (descriptor) after the number, removes these
        df["response"] = (
            df["response"]
                .astype(str)             
                .str.extract(r"(\d+)")    
                .astype("Int64") 
        )
        df["trial"] = df.index + 1
        return df.iloc[:50]
    

    def get_image(self, image: str, size: tuple[int, int] = None) -> Image:
        if not size:
            size = (self.image_width, self.image_height)
        path = os.path.join(self.images_folder, image)
        return Image.open(path).resize(size) #.transpose(Image.Transpose.FLIP_LEFT_RIGHT).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    
    
    def get_most_used_images(self) -> list[str]:
        image_counts =  {img: len(count) for (img, count) in self.image_map.items()}
        max_count = max(image_counts.values())
        return [img for img, count in image_counts.items() if count == max_count]
    

    def get_annotated_images(self) -> list[str]:
        return [
            '2017_11162269.jpg',
            '2017_24623373.jpg',
            '2017_28586643.jpg',
            '2017_28690375.jpg',
            '2017_34226637.jpg',
            '2017_37020948.jpg',
            '2017_44808194.jpg',
            '2017_86316196.jpg',
            '2017_87840980.jpg',
            '2017_98830509.jpg'
        ]


    def get_image_dict(self, image: str) -> dict[int, int | str]:
        return self.image_map[image]


    def get_image_trials(self, image: str, resolve_file: bool = True) -> dict[int, pd.DataFrame]:
        if not resolve_file:
            return self.image_map[image]
        return {
            int(sub): self.get_subject_trial(sub, trial) for (sub, trial) in self.image_map[image].items()
        }
    

    def get_image_answers(self, image: str) -> dict[int, pd.DataFrame]:
        return {
            int(sub): self.get_subject_answers(sub) for sub in self.image_map[image]
        }
    
    def get_image_aois(self, image: str) -> AOICenters:
        return self.image_aois[image]
    

    def get_image_ratings(self, image: str) -> dict[int, int]:
        image_dict = self.get_image_dict(image)
        ratings: dict[int, int] = {}
        for (subject, trial) in image_dict.items():
            answers = self.get_subject_answers(subject)
            response = answers[answers["trial"] == trial]["response"].item()
            ratings[subject] = int(response)
        return ratings
