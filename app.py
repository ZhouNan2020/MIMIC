import gradio as gr
import pandas as pd
import numpy as np
# 导入pickle模块
import pickle
import matplotlib.pyplot as plt
# 导入C:\MyProject\MIMIC\dvt_diabetes\savemodel\c-curve\00calibrated.pickle
with open(r'C:\\MyProject\\MIMIC\\dvt_diabetes\\savemodel\\c-curve\\00calibrated.pickle', 'rb') as f:
    modelcalibration00 = pickle.load(f)
with open(r'C:\\MyProject\\MIMIC\\dvt_diabetes\\savemodel\\c-curve\\28calibrated.pkl', 'rb') as f:
    modelcalibration28 = pickle.load(f)
with open(r'C:\\MyProject\\MIMIC\\dvt_diabetes\\savemodel\\c-curve\\60calibrated.pkl', 'rb') as f:
    modelcalibration60 = pickle.load(f)
with open(r'C:\\MyProject\\MIMIC\\dvt_diabetes\\savemodel\\c-curve\\90calibrated.pkl', 'rb') as f:
    modelcalibration90 = pickle.load(f)

def plot_probabilities(probabilities):
    plt.figure()
    plt.xticks([0, 28, 60, 90])
    plt.plot([0, 28, 60, 90], probabilities, marker='o')
    plt.xlabel('Days')
    plt.ylabel('Probability')
    plt.title('Probability Line Chart')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    return plt.gcf()

def get_risk_category(calibrated_class):
    if calibrated_class == 1:
        return 'Risks that need attention'
    else:
        return 'Lower risk'

def process_input(age_input, height_input, weight_input, lods_input, apache_input, cci_input, oasis_input, saps_input,  sofa_input, alp_max, alp_min, alt_max, alt_min, ast_max, ast_min, bilirubin_max, bilirubin_min, bun_max, bun_min, creatinine_max, creatinine_min, glucose_mean, glucose_min, inr_max, inr_min, pt_max, pt_min, ptt_max, ptt_min, wbc_max, wbc_min, platelet_max, platelet_min):
    # 对lods_input执行标准化，均值为4.99065420560747，标准差为2.92882867084736
    LODS = lods_input 
    # 对age_input执行标准化，均值为68.6822429906542，标准差为12.561298298873
    admission_age = age_input 
    alp_max = alp_max
    alp_min = alp_min
    alt_max = alt_max
    alt_min = alt_min
    apsiii = apache_input
    ast_max = ast_max
    ast_min = ast_min
    bilirubin_total_max = bilirubin_max
    bilirubin_total_min = bilirubin_min
    bun_max = bun_max
    bun_min = bun_min
    charlson_comorbidity_index = cci_input
    creatinine_max = creatinine_max
    creatinine_min = creatinine_min
    glucose_mean = glucose_mean
    glucose_min = glucose_min
    height = height_input
    inr_max = inr_max
    inr_min = inr_min
    oasis = oasis_input
    platelets_max = platelet_max
    platelets_min = platelet_min
    pt_max = pt_max
    pt_min = pt_min
    ptt_max = ptt_max
    ptt_min = ptt_min
    sapsii = saps_input
    sofa_24hours = sofa_input
    wbc_max = wbc_max
    wbc_min = wbc_min
    weight = weight_input

    data_28 = np.array([
        [charlson_comorbidity_index, oasis, LODS, weight, sapsii, apsiii, sofa_24hours, wbc_max, wbc_min, pt_min, inr_min, pt_max, inr_max, admission_type_2, creatinine_min, admission_age, height, platelets_min, ptt_max, admission_location_1, alt_min, sirs, bun_max, insurance_2, bun_min, ptt_min, platelets_max, alp_min, ast_min, bilirubin_total_max, creatinine_max, glucose_max, insurance_1, glucose_mean, glucose_min, alp_max, ast_max, bilirubin_total_min, alt_max, marital_status_1]
    ])
    data_60 = np.array([
        [charlson_comorbidity_index, oasis, apsiii, admission_age, wbc_max, weight, wbc_min, sofa_24hours, sapsii, LODS, height, ptt_min, ptt_max, glucose_min, bun_min, pt_min, glucose_mean, pt_max, bun_max, creatinine_min, inr_min, insurance_2, platelets_min, platelets_max, sirs, glucose_max, creatinine_max, inr_max, ast_max, alt_min, alp_max, ast_min, alp_min, alt_max, bilirubin_total_max, admission_type_5, bilirubin_total_min, insurance_1, marital_status_1, admission_location_1]
    ])
    data_90 = np.array([
        [charlson_comorbidity_index, oasis, apsiii, sapsii, sofa_24hours, wbc_max, admission_age, LODS, height, weight, wbc_min, ptt_max, glucose_min, ptt_min, alp_min, pt_max, alp_max, bun_max, bun_min, creatinine_max, creatinine_min, platelets_max, pt_min, inr_max, bilirubin_total_min, platelets_min, glucose_mean, bilirubin_total_max, glucose_max, insurance_1, inr_min, sirs, alt_min, ast_min, insurance_2, ast_max, alt_max]
    ])
    data_00 = np.array([
        [oasis, sofa_24hours, charlson_comorbidity_index, apsiii, LODS, weight, wbc_max, admission_age, wbc_min, pt_min, insurance_1, creatinine_min, creatinine_max, sapsii, bun_min, inr_min, alp_min, platelets_min, pt_max, platelets_max, marital_status_1, bun_max, insurance_2, height, inr_max, ptt_min, alp_max, sirs, alt_max, alt_min, glucose_min, glucose_mean, ast_max, bilirubin_total_max, race_16, ast_min, ptt_max, admission_type_2, admission_type_5, glucose_max, marital_status_2]
    ])

    # 将数据直接输入概率校准模型
    calibrated_prob_28 = modelcalibration28.predict_proba(data_28)[:, 1]
    calibrated_class_28 = modelcalibration28.predict(data_28)[0]
    
    calibrated_prob_60 = modelcalibration60.predict_proba(data_60)[:, 1]
    calibrated_class_60 = modelcalibration60.predict(data_60)[0]
    
    calibrated_prob_90 = modelcalibration90.predict_proba(data_90)[:, 1]
    calibrated_class_90 = modelcalibration90.predict(data_90)[0]
    
    calibrated_prob_00 = modelcalibration00.predict_proba(data_00)[:, 1]
    calibrated_class_00 = modelcalibration00.predict(data_00)[0]
    
    probabilities = [calibrated_prob_00, calibrated_prob_28, calibrated_prob_60, calibrated_prob_90]
    plot = plot_probabilities(probabilities)
    
    risk_category_00 = get_risk_category(calibrated_class_00)
    risk_category_28 = get_risk_category(calibrated_class_28)
    risk_category_60 = get_risk_category(calibrated_class_60)
    risk_category_90 = get_risk_category(calibrated_class_90)
    # 返回校准后的概率和类别
    return risk_category_00, risk_category_28, risk_category_60, risk_category_90, plot



with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            # 输入控件
            age_input = gr.Number(label="admission_age", step=1)
            height_input = gr.Number(label="Height (cm)", step=0.1)  # 身高输入框，单位为厘米
            weight_input = gr.Number(label="Weight (kg)", step=0.1)  # 体重输入框，单位为千克
            
            # 临床评分部分
            gr.Markdown("## Clinical Scoring Section")
            lods_input = gr.Number(label="LODS (Logistic Organ Dysfunction Score)", step=0.1)
            apache_input = gr.Number(label="APACHE III (Acute Physiology and Chronic Health Evaluation III)", step=0.1)
            cci_input = gr.Number(label="CCI (Charlson Comorbidity Index)", step=0.1)
            oasis_input = gr.Number(label="OASIS (Oxford Acute Severity of Illness Score)", step=0.1)
            saps_input = gr.Number(label="SAPS II (Simplified Acute Physiology Score II)", step=0.1)
            
            sofa_input = gr.Number(label="SOFA (Sequential Organ Failure Assessment, 24hr average)", step=0.1)
        with gr.Column():
            gr.Markdown("## Laboratory indicators on the first day")
            # 血液学检验部分
            gr.Markdown("### Hematologic Tests")
            # 白细胞计数
            gr.Markdown("#### White Blood Cell Count (WBC) (10^9/L)")
            wbc_max = gr.Number(label="Max")
            wbc_min = gr.Number(label="Min")
            # 血小板计数
            gr.Markdown("#### Platelet Count (10^9/L)")
            platelet_max = gr.Number(label="Max")
            platelet_min = gr.Number(label="Min")

            # 肝功能测试部分
            gr.Markdown("### Liver Function Tests")
            # 碱性磷酸酶
            gr.Markdown("#### Alkaline Phosphatase (ALP) (U/L)")
            alp_max = gr.Number(label="Max")
            alp_min = gr.Number(label="Min")
            # 丙氨酸氨基转移酶
            gr.Markdown("#### Alanine Aminotransferase (ALT) (U/L)")
            alt_max = gr.Number(label="Max")
            alt_min = gr.Number(label="Min")
            # 阿斯巴甜氨基转移酶
            gr.Markdown("#### Aspartate Aminotransferase (AST) (U/L)")
            ast_max = gr.Number(label="Max")
            ast_min = gr.Number(label="Min")
            # 总胆红素
            gr.Markdown("#### Total Bilirubin (mg/dL)")
            bilirubin_max = gr.Number(label="Max")
            bilirubin_min = gr.Number(label="Min")

            # 肾功能测试部分
            gr.Markdown("### Renal Function Tests")
            # 尿素氮
            gr.Markdown("#### Blood Urea Nitrogen (BUN) (mg/dL)")
            bun_max = gr.Number(label="Max")
            bun_min = gr.Number(label="Min")
            # 肌酐
            gr.Markdown("#### Creatinine (mg/dL)")
            creatinine_max = gr.Number(label="Max")
            creatinine_min = gr.Number(label="Min")

            # 血糖水平
            gr.Markdown("### Glucose Levels")
            # 血糖
            gr.Markdown("#### Glucose (mg/dL)")
            glucose_mean = gr.Number(label="Mean")
            glucose_min = gr.Number(label="Min")

            # 凝血测试
            gr.Markdown("### Coagulation Tests")
            # 国际标准化比率
            gr.Markdown("#### International Normalized Ratio (INR) (ratio)")
            inr_max = gr.Number(label="Max")
            inr_min = gr.Number(label="Min")
            # 凝血酶原时间
            gr.Markdown("#### Prothrombin Time (PT) (seconds)")
            pt_max = gr.Number(label="Max")
            pt_min = gr.Number(label="Min")
            # 部分凝血活酶时间
            gr.Markdown("#### Partial Thromboplastin Time (PTT) (seconds)")
            ptt_max = gr.Number(label="Max")
            ptt_min = gr.Number(label="Min")

            submit_button = gr.Button("Submit")

        with gr.Column():
            gr.Markdown("Risk Prediction Results")
            output_class_00 = gr.Label(label="00-day Prediction Category")
            #output_prob_28 = gr.Textbox(label="28-day Calibrated Probability", interactive=False)
            output_class_28 = gr.Label(label="28-day Prediction Category")
            #output_prob_60 = gr.Textbox(label="60-day Calibrated Probability", interactive=False)
            output_class_60 = gr.Label(label="60-day Prediction Category")
            #output_prob_90 = gr.Textbox(label="90-day Calibrated Probability", interactive=False)
            output_class_90 = gr.Label(label="90-day Prediction Category")
            #output_prob_00 = gr.Textbox(label="00-day Calibrated Probability", interactive=False)

            output_plot = gr.Plot(label="Probability Line Chart")

    
    submit_button.click(
    process_input,
    inputs=[
        age_input, height_input, weight_input,
        
        lods_input, apache_input, cci_input, oasis_input, saps_input, sofa_input,
        wbc_max, wbc_min, platelet_max, platelet_min,
        alp_max, alp_min, alt_max, alt_min, ast_max, ast_min, bilirubin_max, bilirubin_min,
        bun_max, bun_min, creatinine_max, creatinine_min,
        glucose_mean, glucose_min,
        inr_max, inr_min, pt_max, pt_min, ptt_max, ptt_min
    ],
    outputs=[
        risk_category_00,
        risk_category_28,
        risk_category_60,
        risk_category_90,
        plot
    ]
)
                        
demo.launch()
