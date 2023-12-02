import gradio as gr
import pandas as pd
import numpy as np
# 导入pickle模块
import pickle
import matplotlib.pyplot as plt
# 导入C:\MyProject\MIMIC\dvt_diabetes\savemodel\c-curve\00calibrated.pickle
with open(r'./model/00calibrated_clf.pickle', 'rb') as f:
    modelcalibration00 = pickle.load(f)
with open(r'./model/28calibrated_clf.pickle', 'rb') as f:
    modelcalibration28 = pickle.load(f)
with open(r'./model/60calibrated_clf.pickle', 'rb') as f:
    modelcalibration60 = pickle.load(f)
with open(r'./model/90calibrated_clf.pickle', 'rb') as f:
    modelcalibration90 = pickle.load(f)


def plot_probabilities(probabilities):
    plt.figure(figsize=(18, 11))
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
        [charlson_comorbidity_index, oasis, apsiii, weight, sapsii, ast_max, ast_min, alt_min, admission_age, sofa_24hours, pt_min, LODS, alt_max, wbc_min, bilirubin_total_max, height, wbc_max, inr_min, ptt_min, pt_max, inr_max, creatinine_max]
    ])
    data_60 = np.array([
        [charlson_comorbidity_index, sofa_24hours, wbc_min, apsiii, sapsii, admission_age, wbc_max, alp_min, weight, height, ast_max, oasis, ptt_max, alt_max, LODS, alt_min, ptt_min, alp_max, pt_min, platelets_max, glucose_min, bilirubin_total_max, bun_max, bun_min, bilirubin_total_min, pt_max]
    ])
    data_90 = np.array([
        [charlson_comorbidity_index, apsiii, admission_age, sapsii, LODS, sofa_24hours, oasis, wbc_max, height, wbc_min, weight, bun_min, creatinine_max, ptt_min, ptt_max, creatinine_min, glucose_min, glucose_mean, bilirubin_total_max, alt_min, platelets_min, pt_min, alp_min, bun_max, inr_min, alp_max, platelets_max]
    ])
    data_00 = np.array([
        [oasis, charlson_comorbidity_index, LODS, sofa_24hours, wbc_min, apsiii, sapsii, weight, platelets_min, wbc_max, admission_age, pt_min, alp_min, bun_min, ptt_min, alp_max, inr_min, ast_min]
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
    output_plot = plot_probabilities(probabilities)
    
    risk_category_00 = get_risk_category(calibrated_class_00)
    risk_category_28 = get_risk_category(calibrated_class_28)
    risk_category_60 = get_risk_category(calibrated_class_60)
    risk_category_90 = get_risk_category(calibrated_class_90)
    # 返回校准后的概率和类别
    return risk_category_00, risk_category_28, risk_category_60, risk_category_90, output_plot



with gr.Blocks() as demo:
    with gr.Row():
        gr.Markdown('# Prediction of Death Events in ICU Patients with DVT &Diabetes')
    with gr.Row():
        with gr.Column():
            gr.Markdown('## Basic Information')
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
        
            gr.Markdown("## Laboratory indicators on the first day")
            # 血液学检验部分
            gr.Markdown("### Hematologic Tests")
            # 白细胞计数
            gr.Markdown("#### White Blood Cell Count (WBC) (10^9/L)")
            wbc_max = gr.Number(label="Max")
            wbc_min = gr.Number(label="Min")
        with gr.Column():
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
        with gr.Column():
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
    with gr.Row():
            submit_button = gr.Button("Submit")

    with gr.Row():
            gr.Markdown("## Risk Prediction Results")
    with gr.Row():
            risk_category_00 = gr.Label(label="00-day Prediction Category")
            #output_prob_28 = gr.Textbox(label="28-day Calibrated Probability", interactive=False)
            risk_category_28 = gr.Label(label="28-day Prediction Category")
            #output_prob_60 = gr.Textbox(label="60-day Calibrated Probability", interactive=False)
            risk_category_60 = gr.Label(label="60-day Prediction Category")
            #output_prob_90 = gr.Textbox(label="90-day Calibrated Probability", interactive=False)
            risk_category_90 = gr.Label(label="90-day Prediction Category")
            #output_prob_00 = gr.Textbox(label="00-day Calibrated Probability", interactive=False)
    with gr.Row():
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
        output_plot
    ]
)
                        
demo.launch(share=True)