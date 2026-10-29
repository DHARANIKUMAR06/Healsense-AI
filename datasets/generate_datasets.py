import os
import numpy as np
import pandas as pd

def main():
    np.random.seed(42)
    os.makedirs("datasets", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    n_samples = 1200
    
    # 1. DIABETES
    pregnancies = np.random.poisson(lam=3, size=n_samples).clip(0, 17)
    glucose = np.random.normal(loc=120, scale=30, size=n_samples).clip(50, 200)
    blood_pressure = np.random.normal(loc=70, scale=12, size=n_samples).clip(40, 130)
    skin_thickness = np.random.normal(loc=20, scale=10, size=n_samples).clip(0, 99)
    insulin = np.random.normal(loc=80, scale=40, size=n_samples).clip(0, 840)
    bmi = np.random.normal(loc=32, scale=6, size=n_samples).clip(15, 60)
    pedigree = np.random.exponential(scale=0.4, size=n_samples).clip(0.08, 2.4)
    age = np.random.randint(21, 81, size=n_samples)

    logits = (
        0.035 * (glucose - 100) +
        0.06 * (bmi - 25) +
        0.04 * (age - 30) +
        1.2 * pedigree +
        0.1 * pregnancies - 3.2
    )
    probs = 1 / (1 + np.exp(-logits))
    outcome = (np.random.uniform(0, 1, size=n_samples) < probs).astype(int)

    df_diabetes = pd.DataFrame({
        'Pregnancies': pregnancies,
        'Glucose': np.round(glucose, 1),
        'BloodPressure': np.round(blood_pressure, 1),
        'SkinThickness': np.round(skin_thickness, 1),
        'Insulin': np.round(insulin, 1),
        'BMI': np.round(bmi, 1),
        'DiabetesPedigreeFunction': np.round(pedigree, 3),
        'Age': age,
        'Outcome': outcome
    })
    df_diabetes.to_csv('datasets/diabetes.csv', index=False)

    # 2. HEART
    age_h = np.random.randint(29, 78, size=n_samples)
    sex_h = np.random.binomial(1, 0.65, size=n_samples)
    cp_h = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.47, 0.16, 0.28, 0.09])
    trestbps_h = np.random.normal(loc=131, scale=17, size=n_samples).clip(94, 200)
    chol_h = np.random.normal(loc=246, scale=50, size=n_samples).clip(126, 564)
    fbs_h = np.random.binomial(1, 0.15, size=n_samples)
    restecg_h = np.random.choice([0, 1, 2], size=n_samples, p=[0.49, 0.48, 0.03])
    thalach_h = np.random.normal(loc=149, scale=22, size=n_samples).clip(71, 202)
    exang_h = np.random.binomial(1, 0.32, size=n_samples)
    oldpeak_h = np.random.exponential(scale=1.0, size=n_samples).clip(0.0, 6.2)
    slope_h = np.random.choice([0, 1, 2], size=n_samples, p=[0.07, 0.46, 0.47])
    ca_h = np.random.choice([0, 1, 2, 3, 4], size=n_samples, p=[0.58, 0.21, 0.12, 0.07, 0.02])
    thal_h = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.01, 0.06, 0.54, 0.39])

    logits_h = (
        0.03 * (age_h - 50) +
        0.8 * sex_h -
        0.9 * cp_h +
        0.015 * (trestbps_h - 120) +
        0.005 * (chol_h - 200) -
        0.03 * (thalach_h - 150) +
        1.1 * exang_h +
        0.6 * oldpeak_h +
        0.7 * ca_h +
        0.8 * (thal_h == 3).astype(int) - 1.2
    )
    probs_h = 1 / (1 + np.exp(-logits_h))
    target_h = (np.random.uniform(0, 1, size=n_samples) < probs_h).astype(int)

    df_heart = pd.DataFrame({
        'age': age_h,
        'sex': sex_h,
        'cp': cp_h,
        'trestbps': np.round(trestbps_h, 1),
        'chol': np.round(chol_h, 1),
        'fbs': fbs_h,
        'restecg': restecg_h,
        'thalach': np.round(thalach_h, 1),
        'exang': exang_h,
        'oldpeak': np.round(oldpeak_h, 2),
        'slope': slope_h,
        'ca': ca_h,
        'thal': thal_h,
        'target': target_h
    })
    df_heart.to_csv('datasets/heart.csv', index=False)

    # 3. KIDNEY
    age_k = np.random.randint(18, 91, size=n_samples)
    bp_k = np.random.choice([50, 60, 70, 80, 90, 100, 110, 120, 180], size=n_samples)
    sg_k = np.random.choice([1.005, 1.010, 1.015, 1.020, 1.025], size=n_samples)
    al_k = np.random.choice([0, 1, 2, 3, 4, 5], size=n_samples, p=[0.5, 0.2, 0.15, 0.08, 0.05, 0.02])
    su_k = np.random.choice([0, 1, 2, 3, 4, 5], size=n_samples, p=[0.7, 0.15, 0.08, 0.04, 0.02, 0.01])
    rbc_k = np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2])
    pc_k = np.random.choice([0, 1], size=n_samples, p=[0.75, 0.25])
    pcc_k = np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])
    ba_k = np.random.choice([0, 1], size=n_samples, p=[0.93, 0.07])
    bgr_k = np.random.normal(loc=148, scale=75, size=n_samples).clip(70, 490)
    bu_k = np.random.normal(loc=57, scale=50, size=n_samples).clip(10, 390)
    sc_k = np.random.normal(loc=3.0, scale=3.5, size=n_samples).clip(0.4, 15.0)
    sod_k = np.random.normal(loc=137, scale=8, size=n_samples).clip(110, 150)
    pot_k = np.random.normal(loc=4.6, scale=1.2, size=n_samples).clip(2.5, 7.5)
    hemo_k = np.random.normal(loc=12.5, scale=2.8, size=n_samples).clip(3.1, 17.8)

    logits_k = (
        0.02 * (age_k - 40) +
        0.015 * (bp_k - 80) +
        1.2 * al_k +
        0.5 * su_k +
        1.1 * (sg_k <= 1.015).astype(int) +
        0.8 * rbc_k +
        0.7 * pc_k +
        0.8 * sc_k +
        0.03 * bu_k -
        0.6 * (hemo_k - 10) - 2.5
    )
    probs_k = 1 / (1 + np.exp(-logits_k))
    classification_k = (np.random.uniform(0, 1, size=n_samples) < probs_k).astype(int)

    df_kidney = pd.DataFrame({
        'age': age_k,
        'bp': bp_k,
        'sg': sg_k,
        'al': al_k,
        'su': su_k,
        'rbc': rbc_k,
        'pc': pc_k,
        'pcc': pcc_k,
        'ba': ba_k,
        'bgr': np.round(bgr_k, 1),
        'bu': np.round(bu_k, 1),
        'sc': np.round(sc_k, 2),
        'sod': np.round(sod_k, 1),
        'pot': np.round(pot_k, 2),
        'hemo': np.round(hemo_k, 1),
        'classification': classification_k
    })
    df_kidney.to_csv('datasets/kidney.csv', index=False)

    # 4. MALARIA
    wbc_m = np.random.normal(loc=7500, scale=2500, size=n_samples).clip(2500, 18000)
    rbc_m = np.random.normal(loc=4.5, scale=0.8, size=n_samples).clip(2.0, 6.5)
    plt_m = np.random.normal(loc=200000, scale=80000, size=n_samples).clip(20000, 450000)
    hemo_m = np.random.normal(loc=13.0, scale=2.2, size=n_samples).clip(5.5, 17.5)
    fever_dur_m = np.random.randint(1, 15, size=n_samples)
    temp_m = np.random.normal(loc=99.8, scale=1.8, size=n_samples).clip(97.5, 105.0)
    chills_m = np.random.binomial(1, 0.4, size=n_samples)
    spleen_m = np.random.binomial(1, 0.25, size=n_samples)

    logits_m = (
        -0.00003 * (plt_m - 150000) -
        0.35 * (hemo_m - 12) +
        0.25 * fever_dur_m +
        0.9 * (temp_m - 99.0) +
        1.4 * chills_m +
        1.6 * spleen_m - 3.8
    )
    probs_m = 1 / (1 + np.exp(-logits_m))
    target_m = (np.random.uniform(0, 1, size=n_samples) < probs_m).astype(int)

    df_malaria = pd.DataFrame({
        'wbc_count': np.round(wbc_m, 0),
        'rbc_count': np.round(rbc_m, 2),
        'platelet_count': np.round(plt_m, 0),
        'hemoglobin': np.round(hemo_m, 1),
        'fever_duration': fever_dur_m,
        'body_temperature': np.round(temp_m, 1),
        'chills_rigor': chills_m,
        'spleen_enlargement': spleen_m,
        'target': target_m
    })
    df_malaria.to_csv('datasets/malaria.csv', index=False)

    # 5. DENGUE
    wbc_d = np.random.normal(loc=6500, scale=2800, size=n_samples).clip(1500, 15000)
    plt_d = np.random.normal(loc=180000, scale=85000, size=n_samples).clip(10000, 400000)
    hct_d = np.random.normal(loc=41.0, scale=5.5, size=n_samples).clip(28.0, 58.0)
    ns1_d = np.random.binomial(1, 0.3, size=n_samples)
    igg_d = np.random.binomial(1, 0.25, size=n_samples)
    igm_d = np.random.binomial(1, 0.35, size=n_samples)
    fever_dur_d = np.random.randint(1, 11, size=n_samples)
    temp_d = np.random.normal(loc=100.2, scale=1.6, size=n_samples).clip(98.0, 105.0)

    logits_d = (
        -0.0004 * (wbc_d - 5000) -
        -0.000035 * (plt_d - 100000) +
        0.2 * (hct_d - 40) +
        2.5 * ns1_d +
        1.2 * igm_d +
        0.8 * igg_d +
        0.6 * (temp_d - 99.5) - 3.2
    )
    probs_d = 1 / (1 + np.exp(-logits_d))
    target_d = (np.random.uniform(0, 1, size=n_samples) < probs_d).astype(int)

    df_dengue = pd.DataFrame({
        'wbc_count': np.round(wbc_d, 0),
        'platelet_count': np.round(plt_d, 0),
        'hematocrit': np.round(hct_d, 1),
        'ns1_antigen': ns1_d,
        'igg': igg_d,
        'igm': igm_d,
        'fever_duration': fever_dur_d,
        'body_temperature': np.round(temp_d, 1),
        'target': target_d
    })
    df_dengue.to_csv('datasets/dengue.csv', index=False)
    print("All 5 datasets successfully generated in datasets/")

if __name__ == '__main__':
    main()
