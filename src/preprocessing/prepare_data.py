import pandas as pd


def prepare_data(input_path, output_path):
    df = pd.read_csv(input_path)

    # =========================
    # VARIABLE OBJETIVO
    # =========================
    df["Diabetes_binary"] = df["Diabetes_binary"].map({
        0: "No Diabetes",
        1: "Diabetes"
    })

    # =========================
    # SEXO
    # =========================
    df["Sex"] = df["Sex"].map({
        0: "Female",
        1: "Male"
    })

    # =========================
    # VARIABLES BINARIAS
    # =========================
    binary_cols = [
        "HighBP", "HighChol", "CholCheck", "Smoker",
        "Stroke", "HeartDiseaseorAttack", "PhysActivity",
        "Fruits", "Veggies", "HvyAlcoholConsump",
        "AnyHealthcare", "NoDocbcCost", "DiffWalk"
    ]

    for col in binary_cols:
        df[col] = df[col].map({0: "No", 1: "Yes"})

    # =========================
    # SALUD GENERAL
    # =========================
    genhlth_map = {
        1: "Excellent",
        2: "Very Good",
        3: "Good",
        4: "Fair",
        5: "Poor"
    }
    df["GenHlth"] = df["GenHlth"].map(genhlth_map)

    # =========================
    # EDAD (IMPORTANTE)
    # =========================
    df["AgeGroupNumeric"] = df["Age"]

    age_map = {
        1: "18-24", 2: "25-29", 3: "30-34", 4: "35-39",
        5: "40-44", 6: "45-49", 7: "50-54", 8: "55-59",
        9: "60-64", 10: "65-69", 11: "70-74",
        12: "75-79", 13: "80+"
    }
    df["Age"] = df["Age"].map(age_map)

    # =========================
    # EDUCACIÓN
    # =========================
    education_map = {
        1: "No school",
        2: "Elementary",
        3: "Some high school",
        4: "High school",
        5: "Some college",
        6: "College graduate"
    }
    df["Education"] = df["Education"].map(education_map)

    # =========================
    # INGRESO
    # =========================
    income_map = {
        1: "<10k", 2: "10k-15k", 3: "15k-20k",
        4: "20k-25k", 5: "25k-35k", 6: "35k-50k",
        7: "50k-75k", 8: ">75k"
    }
    df["Income"] = df["Income"].map(income_map)

    # =========================
    # GUARDAR
    # =========================
    df.to_csv(output_path, index=False)

    return df


if __name__ == "__main__":
    prepare_data(
        input_path="data/raw/diabetes_raw.csv",
        output_path="data/processed/diabetes_clean.csv"
    )