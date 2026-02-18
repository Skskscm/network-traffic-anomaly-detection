# AI-Based Network Traffic Anomaly Detection

This repository contains an implementation of an unsupervised anomaly detection system for identifying abnormal network traffic patterns indicative of Distributed Denial-of-Service (DDoS) attacks.

This implementation was developed independently as part of a course mini-project.

## Methodology

A Variational Autoencoder (VAE) was trained on time-series sequences representing normal network traffic behaviour. The trained model attempts to reconstruct input traffic patterns.

Anomalies are detected using reconstruction error between the original and reconstructed traffic sequences. Traffic samples with reconstruction error above a calibrated threshold are flagged as anomalous.

## Dataset

The model was trained and evaluated on synthetically generated network traffic data with injected attack patterns.

## Limitation

This model has been evaluated on synthetic data only and has not yet been tested on publicly available intrusion detection datasets.
