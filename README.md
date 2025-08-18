# Gaze3P analysis

## Notes

- in the edf folder file 17 doesnt exist, file 33 is corrupted

## Project structure

`eyelink_raw`: contains the data provided by the original paper, plus the events from the edf files extracted as ascii files (no changes in content) 

`eyelink_data`:

 ```
Project
|-- eyelink_raw
|   |-- edf
|   |   |-- sub_1.edf
|   |   |-- sub_2.edf
|   |   |-- ...
|   |   |-- sub_102.edf
|   |
|   |-- asc
|       |-- sub_1.asc
|       |-- sub_2.asc
|       |-- ...
|       |-- sub_102.asc
|
|-- eyelink_data
    |-- answers
    |   |-- sub_1.edf
    |   |-- sub_2.edf
    |   |-- ...
    |   |-- sub_102.edf
    |
    |-- images
    |   |-- sub_1.edf
    |   |-- sub_2.edf
    |   |-- ...
    |   |-- sub_102.edf
    |
    |-- scanpaths
        |-- sub_1
        |    |-- images.json
        |    |-- trial_1.csv
        |    |-- trial_2.csv
        |
        |-- sub_2
             |
             |-- images.json
             |-- trial_1.csv
             |-- trial_2.csv

    

```







































---
from the repo:

## Experiment Specifications

### Software and Hardware specifications

1. Software used: Opensesame
2. Windows OS
3. EyeTracker: EyeLink 1000 Plus    


### Dataset Specifications

1. The dataset used: Validation Set and Test Set of VISPR dataset - https://tribhuvanesh.github.io/vpa/
    - Labels - Safe vs 68 Categories of Private Images


2. Private Attribute Selection

We require 25 primary attributes to sort from. We are selecting the most sensitive ones. Discarding nudity and semi-nudity labels all together as primary attribute.

Selecting Criteria: Every block will have something in writing and faces.

Number of Attributes per Block: 5

1. **Block 1**:  Fingerprint, Receipts, Occupation, Sexual Orientation, Political Opinion - (Personal Description, Documents, Employment, Personal Life, Personal Life)

2. **Block 2**: Signature, Tickets, Medical Treatment, Personal Occasion, Home address (Complete) - ( Personal Information, Documents, Health, Personal Life, Whereabouts)

3. **Block 3**:  Race, Passports, Online Conversations, Social Circle, Phone Number - (Personal Information, Documents, Internet, Relationship, Personal Life)

4. **Block 4**: Full Name, Mail, License Plate Complete, Personal Relationship, Visited Location (Partial) - (Personal Information, Documents, Automobile, Personal Life, Whereabouts)

5. **Block 5**: Face Complete, Credit Card, Medical History, Email Content, Religion - (Personal Description, Documents, Health, Internet, Personal Life)


This attributes names are referred from the attributes.tsv file in the dataset_generation_scripts folder.
