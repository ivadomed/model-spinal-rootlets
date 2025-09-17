import argparse
import pandas as pd
import os

def get_parser():
    """
    Function to parse command line arguments.
    :return: command line arguments
    """
    parser = argparse.ArgumentParser(description='Generate a Bland-Altman and correlation plot in one figure.')
    parser.add_argument('-i', required=True, type=str, help='Path to the CSV file with distances from PMJ.')
    return parser

def process_data(participants_path):
    """
    """

    # Merge data with participant's data (the height will be used for normalization)
    df_participants = pd.read_csv(participants_path, sep="\t")

    # calculate how many males (m, M) and females (f, F) are in the dataset (column sex)
    num_males = df_participants["sex"].str.lower().value_counts().get('m', 0)
    num_females = df_participants["sex"].str.lower().value_counts().get('f', 0)
    num_not_known = df_participants["sex"].str.lower().value_counts().get('n/a', 0)

    num_males_percent = num_males/(len(df_participants)) * 100
    num_females_percent = num_females/(len(df_participants)) * 100
    num_not_known_percent = num_not_known/(len(df_participants)) * 100

    # Compute age statistics by sex
    average_age_male = df_participants[df_participants['sex'] == 'm']['age'].mean()
    sd_age_male = df_participants[df_participants['sex'] == 'm']['age'].std()

    average_age_female = df_participants[df_participants['sex'] == 'f']['age'].mean()
    sd_age_female = df_participants[df_participants['sex'] == 'f']['age'].std()

    overall_age_mean = df_participants['age'].mean()
    overall_age_std = df_participants['age'].std()

    # Print results
    print(f"Number of males: {num_males} ({num_males_percent:.1f}%)")
    print(f"Number of females: {num_females} ({num_females_percent:.1f}%)")
    print(f"Number of not known: {num_not_known} ({num_not_known_percent:.1f}%)")
    print(f"Average age (males): {average_age_male:.2f} ± {sd_age_male:.2f}")
    print(f"Average age (females): {average_age_female:.2f} ± {sd_age_female:.2f}")
    print(f"Overall average age: {overall_age_mean:.2f} ± {overall_age_std:.2f}")


def main():
    parser = get_parser()
    args = parser.parse_args()

    # get the variables from the command line
    participants_path = os.path.abspath(args.i)

    # Process the data for spinal and vertebral levels
    df = process_data(participants_path)



if __name__ == "__main__":
    main()
