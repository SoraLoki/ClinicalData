FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime
WORKDIR /app

# Install git, then Python dependencies (including CRF from GitHub)
RUN apt-get update && apt-get install -y git \
 && python3 -m pip install --upgrade pip \
 && pip install pandas numpy scikit-learn transformers matplotlib protobuf seaborn statsmodels\
 && pip install git+https://github.com/kmkurn/pytorch-crf.git

COPY FirstTry_numberOfAnnotations.py .
COPY nbme-score-clinical-patient-notes .
COPY HyperParameter_ClinicalPatientNotes.py .
COPY SCPN_PLOTS.py .
COPY Bert_QA.py .
COPY SCPN_PLOTS_CFR_F1.py .
COPY FINAL_CFR_LOSS_adjusted.py .
