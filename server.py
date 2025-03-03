from fastapi import FastAPI
import torch
from torch.nn import functional as F
from pydantic import BaseModel
from model import TransformerClassification  # مدل را از `model.py` لود می‌کنیم

app = FastAPI()

# **ستون‌های موردنیاز برای مدل**
column_names = [
    "woman_age", "man_age", "woman_height", "man_height", "woman_weight", "man_weight",
    "woman_bmi", "man_bmi", "tubal_factors", "infertility_duration", "menstrual_disorders",
    "unexplained_infertility", "severe_pelvic_adhesion", "woman_endometriosis", "woman_baseline_fsh",
    "woman_baseline_lh", "baseline_prl", "baseline_amh", "woman_diabetes", "woman_hypertension",
    "man_diabetes", "man_hypertension", "woman_hypothyroidism", "man_hypothyroidism", "woman_anemia",
    "man_anemia", "woman_laparotomy", "cyst_aspiration", "diagnostic_hysteroscopy",
    "woman_therapeutic_laparoscopy", "therapeutic_hysteroscopy", "pco", "hsg_uterine_cavity",
    "hydrosalpinx", "total_afc", "male_factor", "woman_hepatitis", "man_hepatitis",
    "testicular_biopsy", "tese_outcome", "man_karyotype", "man_primary_infertility",
    "woman_primary_infertility", "man_secondary_infertility", "woman_secondary_infertility",
    "retrieved_oocytes_count", "transferred_embryos_count", "embryo_quality", "embryo_freezing_status",
    "embryo_morphology", "embryo_transfer_day", "man_high_risk_job", "woman_high_risk_job",
    "salpingitis", "varicocele_surgery", "mother_smoking_and_opiates", "father_smoking_and_opiates",
    "father_alcohol_consumption", "mother_lupus_and_antiphospholipid_syndrome", "man_covid",
    "woman_covid", "man_covid_vaccination_history", "adenomyosis", "dfi",
    "man_vitamin_d", "woman_vitamin_d", "pap_smear"
]

# **تعریف مدل ورودی برای دریافت داده‌ها از کاربر**
class InputData(BaseModel):
    woman_age: float
    man_age: float
    woman_height: float
    man_height: float
    woman_weight: float
    man_weight: float
    woman_bmi: float
    man_bmi: float
    tubal_factors: int
    infertility_duration: float
    menstrual_disorders: int
    unexplained_infertility: int
    severe_pelvic_adhesion: int
    woman_endometriosis: int
    woman_baseline_fsh: float
    woman_baseline_lh: float
    baseline_prl: float
    baseline_amh: float
    woman_diabetes: int
    woman_hypertension: int
    man_diabetes: int
    man_hypertension: int
    woman_hypothyroidism: int
    man_hypothyroidism: int
    woman_anemia: int
    man_anemia: int
    woman_laparotomy: int
    cyst_aspiration: int
    diagnostic_hysteroscopy: int
    woman_therapeutic_laparoscopy: int
    therapeutic_hysteroscopy: int
    pco: int
    hsg_uterine_cavity: int
    hydrosalpinx: int
    total_afc: float
    male_factor: int
    woman_hepatitis: int
    man_hepatitis: int
    testicular_biopsy: int
    tese_outcome: int
    man_karyotype: int
    man_primary_infertility: int
    woman_primary_infertility: int
    man_secondary_infertility: int
    woman_secondary_infertility: int
    retrieved_oocytes_count: int
    transferred_embryos_count: int
    embryo_quality: int
    embryo_freezing_status: int
    embryo_morphology: int
    embryo_transfer_day: int
    man_high_risk_job: int
    woman_high_risk_job: int
    salpingitis: int
    varicocele_surgery: int
    mother_smoking_and_opiates: int
    father_smoking_and_opiates: int
    father_alcohol_consumption: int
    mother_lupus_and_antiphospholipid_syndrome: int
    man_covid: int
    woman_covid: int
    man_covid_vaccination_history: int
    adenomyosis: int
    dfi: float
    man_vitamin_d: float
    woman_vitamin_d: float
    pap_smear: int

# **بارگذاری مدل**
input_dim = len(column_names)
hidden_dim = input_dim
num_heads = 2
num_layers = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TransformerClassification(input_dim=input_dim, hidden_dim=hidden_dim, num_heads=num_heads, num_layers=num_layers).to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

@app.post("/predict/")
async def predict(data: InputData):
    try:
        # **تبدیل داده‌های ورودی به لیست و سپس `Tensor`**
        input_values = [getattr(data, col) for col in column_names]
        X_tensor = torch.tensor([input_values], dtype=torch.float32).to(device)

        # **پیش‌بینی مدل**
        with torch.no_grad():
            outputs = model(X_tensor)
            predictions = F.softmax(outputs, dim=1)[:, 1]  # احتمال موفقیت (برچسب 1)
            y_pred = (predictions > 0.5).int().cpu().numpy()

        # **محاسبه درصد موفقیت و عدم موفقیت**
        success_rate = predictions.item() * 100  # درصد موفقیت
        failure_rate = 100 - success_rate  # درصد عدم موفقیت

        return {
            "success_rate": round(success_rate, 2),
            "failure_rate": round(failure_rate, 2)
        }
    
    except Exception as e:
        return {"error": str(e)}

