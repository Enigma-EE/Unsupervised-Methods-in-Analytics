### Understanding the Assignment Requirements

Your assignment focuses on three main tasks:

1. Conduct a literature review on anomaly detection methods with an emphasis on clustering techniques.
2. Design an anomaly detection algorithm based on k-means.
3. Apply this algorithm to a given dataset (`a2-housing.csv`) using SAS Enterprise Miner.

The deliverables include a paper, presentation slides, and a recorded presentation. The paper itself is divided into specific sections, including an introduction, literature review, algorithm description, and application of the algorithm to the dataset. The assignment accounts for 35% of the total course marks and is due on November 5, 2023, 11:59 pm Adelaide time.

### Paper Structure

1. **Introduction (10%)**
    - What is Anomaly Detection
    - Motivation and Examples
    - Main categories of anomaly detection methods, including the basic idea of each type of methods and suitable scenarios to use (data types, dimensionality etc.). 


### Anomaly Detection Using Unsupervised Methods vs. Traditional Rule-Based Systems

#### Unsupervised Methods: 

1. **Scalability**: Unsupervised methods like K-means or DBSCAN scale well with the size of the data. 
   
2. **Adaptability**: These methods can adapt to unseen or evolving data patterns, provided that the algorithm is periodically retrained.
  
3. **Complex Relationships**: Capable of capturing complex relationships between variables, including non-linearities.

4. **Minimal Domain Knowledge**: No explicit rules are needed, which is beneficial when domain expertise is lacking.

5. **False Positives**: May generate false positives if the model isn't well-tuned or if the underlying assumptions (like cluster shapes in K-means) don't hold.

6. **Interpretability**: Usually less interpretable than rule-based systems, unless additional steps are taken for feature selection or model explanation.

#### Traditional Rule-Based Systems:

1. **Simplicity**: Rule-based systems are straightforward to implement and understand.

2. **Domain Knowledge**: These systems usually rely heavily on domain expertise to formulate rules.

3. **Lack of Adaptability**: Static rules can quickly become outdated and fail to adapt to new kinds of anomalies unless manually updated.

4. **Scalability**: As the number of rules grows, the system may become computationally expensive to maintain.

5. **Comprehensiveness**: The system is only as good as the rules it contains; it may miss anomalies that have not been anticipated.

6. **Interpretability**: Rule-based systems are highly interpretable because the logic behind the decision-making process is transparent.

#### Comparative Summary:

1. **Flexibility**: Unsupervised methods offer more flexibility in capturing unknown or evolving anomaly patterns, while rule-based systems are rigid but easy to interpret.

2. **Domain Knowledge**: Rule-based systems require extensive domain knowledge; unsupervised methods do not.

3. **Maintenance**: Rule-based systems require continuous manual updates, whereas unsupervised methods require periodic retraining, which can be automated.

4. **Computational Load**: Rule-based systems may have a lower computational load for a small set of rules but can become inefficient as the number of rules grows. In contrast, unsupervised methods are inherently more scalable but may require more computational resources upfront.

In summary, the choice between unsupervised and rule-based methods for anomaly detection largely depends on the specific requirements of the task, including scalability, adaptability, and the availability of domain knowledge.

---




### Introduction

#### Anomaly Detection Overview
- Define anomaly detection and its purpose.
- Illustrate the concept with examples from cybersecurity, finance, etc.
- Introduce the main categories of anomaly detection methods and the scenarios in which they are applicable.

#### Motivation for Anomaly Detection
- Explain the increasing importance of anomaly detection in various fields.
- Discuss the challenges addressed by anomaly detection, including zero-day attack detection.

#### Categories of Anomaly Detection Methods
- Statistical-based methods: Discuss their reliance on statistical models.
- Machine learning-based methods: Detail their learning-based approach to identifying anomalies.


---





### Literature Review of Clustering-Based Anomaly Detection Methods

#### Overview of Clustering-Based Methods
- Present a categorization of clustering-based anomaly detection literature.
- Summarize the basic ideas of the categorized methods.

#### Detailed Review of Methods
- Provide a descriptive analysis of each method type.
- Discuss key papers representing each method type.

#### Comparison and Analysis
- Compare the methods in terms of advantages, disadvantages, and use-cases.
- Discuss the similarities and differences across these methods.

### References
- Use a numbered citation style throughout the document.
- Include a comprehensive reference list at the end of the document.

---

### Literature Review on Clustering-Based Anomaly Detection Methods


[1] Guo, P., Wang, L., Shen, J., & Dong, F. (2021). A Hybrid Unsupervised Clustering-Based Anomaly Detection Method. Tsinghua Science & Technology, 26(2):146-153. DOI: 10.26599/TST.2019.9010051.


- Machine learning-based methods (Refer to Guo et al., 2021 [1]).
  - **Sub-Space Clustering (SSC)**: Suitable for identifying patterns and anomalies in data, particularly network traffic for intrusion detection (Guo et al., 2021 [1]).
  - **One Class Support Vector Machine (OCSVM)**: Trained on normal instances to detect deviations, effective for unknown attack types, including zero-day attacks (Guo et al., 2021 [1]).


- Clustering-based methods are effective for detecting novel attacks without prior knowledge, as demonstrated in the hybrid approach combining SSC and OCSVM (Guo et al., 2021 [1]).


- Discussion on the hybrid method combining SSC and OCSVM (Guo et al., 2021 [1]).
  - Evaluation on the NSL-KDD dataset showcases the method's superiority in performance against existing techniques.


- Advantages of the hybrid SSC and OCSVM method include detection of known and unknown attacks, adaptability, and suitability for real-time intrusion detection (Guo et al., 2021 [1]).



**Literature Review on Clustering-Based Anomaly Detection Methods**

This literature review examines clustering-based anomaly detection methods, categorizing them into types and providing a brief overview of each. It also reviews representative methods within each category, highlighting their key features and contributions. Finally, it offers a comparison of the different types of methods, outlining their advantages, disadvantages, and notable differences.

*Types of Clustering-Based Anomaly Detection Methods:*

1. **Sub-Space Clustering (SSC):** SSC partitions data into subspaces based on similarity, identifying clusters within the data. It captures underlying data structure and patterns.

2. **One Class Support Vector Machine (OCSVM):** OCSVM is an anomaly detection algorithm that defines normal behavior boundaries and classifies instances outside these boundaries as anomalies. It's effective for detecting unknown or novel attacks.

*Review of Representative Methods:*

- *SSC*: SSC is used to identify data subspaces and patterns that may indicate attacks.

- *OCSVM*: OCSVM excels in detecting unknown or novel attacks by learning normal behavior boundaries.

*Comparison of Methods:*

- *Advantages*: Clustering-based methods offer flexibility, adaptability to new attack patterns, reduced reliance on labeled data, detection of subtle anomalies, scalability to handle large data volumes, and the ability to detect emerging threats.

- *Effectiveness*: The combination of SSC and OCSVM in the proposed method, as demonstrated on the NSL-KDD dataset, outperforms existing techniques, making it effective for cyber intrusion detection, especially for known, unknown, and zero-day attacks.




---


[2] Y. Qiu, T. Misu and C. Busso, "Unsupervised Scalable Multimodal Driving Anomaly Detection," in IEEE Transactions on Intelligent Vehicles, vol. 8, no. 4, pp. 3154-3165, April 2023, doi: 10.1109/TIV.2022.3160861.

**Literature Review on Unsupervised Contrastive Driving Anomaly Detection**

This literature review explores unsupervised contrastive methods for driving anomaly detection, focusing on a paper by Y. Qiu, T. Misu, and C. Busso. The review includes an introduction to anomaly detection, its motivation, and examples. It also discusses the main categories of anomaly detection methods, highlighting the basic idea of each type and their suitable scenarios.

*Anomaly Detection:*

- **What is Anomaly Detection:** Anomaly detection is the process of identifying patterns or instances that significantly deviate from the norm or expected behavior within a dataset.
  
- **Motivation and Examples:** Anomaly detection is motivated by the need to identify unusual or abnormal events, objects, or actions that pose risks or indicate problems in various domains. In the context of driving, it helps identify unexpected driving behaviors that can increase accident risks.

*Main Categories of Anomaly Detection Methods:*

- **Supervised Approaches:** These methods effectively identify specific driving anomalies but require labeled data for training. They are suitable when known types of driving anomalies can be labeled.

- **Unsupervised Approaches:** Unsupervised methods, like the proposed approach in the paper, automatically identify unexpected driving scenarios without labeled data. They excel in scenarios with a wide range of possible anomalies and no prior labeling.

- **Scalable Formulation:** The paper also focuses on scalability, allowing the addition of new modalities. This suits scenarios where new data sources or modalities can enhance anomaly detection.

In summary, this concise literature review provides an overview of unsupervised contrastive driving anomaly detection, its motivation, and the suitability of different anomaly detection methods, aligning with the paper's content and your requirements.

---


[3] M. Zhao, R. Furuhata, M. Agung, H. Takizawa and T. Soma, "Failure Prediction in Datacenters Using Unsupervised Multimodal Anomaly Detection," 2020 IEEE International Conference on Big Data (Big Data), Atlanta, GA, USA, 2020, pp. 3545-3549, doi: 10.1109/BigData50022.2020.9378419.


This literature review explores multimodal anomaly detection methods for predicting hard drive failures in datacenters, with a focus on a paper by M. Zhao et al. The review includes an introduction to anomaly detection, its motivation, and examples. It also discusses the main categories of anomaly detection methods, highlighting the importance of multimodal approaches for large-scale datacenter systems.

*Anomaly Detection:*

- **What is Anomaly Detection:** Anomaly detection is the process of identifying patterns or instances that significantly deviate from normal behavior or expected patterns within a dataset.

- **Motivation and Examples:** Anomaly detection is motivated by the need to proactively prevent failures and minimize downtime in datacenters. Examples include predicting hard drive failures, detecting fraudulent transactions, and monitoring network intrusions.

*Main Categories of Anomaly Detection Methods:*

- **Threshold-based Methods:** These methods use predefined thresholds to classify instances as normal or anomalous. They have limitations in large-scale datacenter systems, where setting optimal thresholds can be challenging.

- **Multimodal Methods:** Multimodal anomaly detection integrates data from various sensors to provide a comprehensive understanding of system behavior. It is particularly important for large-scale datacenters, where different sensor types offer complementary insights.

*Importance of Multimodal Anomaly Detection in Datacenters:*

- Multimodal anomaly detection is crucial for large-scale datacenters because it combines data from diverse sensors, improving accuracy and comprehensiveness in understanding system behavior.

- Conventional threshold-based methods, which consider each sensor independently, may not be sufficient for large datacenters due to the difficulty of setting optimal thresholds.

- Multimodal approaches capture various aspects of system behavior, including temporal and spatial anomalies, leading to enhanced detection capabilities.

- Multimodal anomaly detection can provide early warnings of failure signs before actual failures occur, as demonstrated in the study.

*Comparison of Multimodal and Unimodal Approaches:*

- The study's unimodal results showed that the auditory and system performance model could detect temporal anomalies, while the thermal model could detect spatial anomalies.

- In contrast, the multimodal approach, even with simple filter and detection algorithms, detected failure signs before actual failures and earlier than the auditory unimodal approach.




---

[4]S. Chen, X. Li and L. Zhao, "Hyperspectral Anomaly Detection with Data Sphering and Unsupervised Target Detection," IGARSS 2022 - 2022 IEEE International Geoscience and Remote Sensing Symposium, Kuala Lumpur, Malaysia, 2022, pp. 1975-1978, doi: 10.1109/IGARSS46834.2022.9884083.


**Main categories of anomaly detection methods used in this paper:**

- **Low rank and sparse representation (LRaSR)-based approaches**: These methods decompose hyperspectral data into low-rank components for background, sparse components for anomalies, and residual components for noise. They are suitable for hyperspectral data with low-rank background and sparse anomalies  .

- **Autoencoder (AE)-based methods**: These methods use neural networks to learn the underlying structure of the data and reconstruct it. Anomalies are detected by measuring the reconstruction error. AE-based methods are suitable for high-dimensional data and can capture complex patterns .

- **Data sphering and unsupervised target detection with sparse cardinality (DS-UTSSC)**: This novel method proposed in the paper combines data sphering, unsupervised target detection, and sparse cardinality. It removes the background using data sphering, generates a potential anomaly component through unsupervised target detection, and incorporates sparse cardinality to reduce noise impact. DS-UTSSC is competitive against LRaSR-based models and AE-based methods in hyperspectral anomaly detection  .

These methods are suitable for hyperspectral data with different characteristics, such as low-rank background, sparse anomalies, and high dimensionality. They offer different approaches to detect anomalies and can be applied in various scenarios depending on the specific data types and dimensionality.


Certainly, I've condensed and refined the information to provide a concise note focusing on the DS-UTSSC method for hyperspectral anomaly detection and its key components:

**Hyperspectral Anomaly Detection Using DS-UTSSC Method**

The DS-UTSSC method is introduced in the paper as an effective approach for hyperspectral anomaly detection. It combines data sphering, unsupervised target detection, and sparse cardinality to improve the accuracy of anomaly detection. Here's a breakdown of its components and their significance:

1. **Data Sphering (DS):** DS is employed to eliminate background noise from the original hyperspectral data, enhancing the precision of anomaly detection.

2. **Unsupervised Target Detection:** This component identifies potential anomalies within the hyperspectral data without relying on labeled training samples or prior knowledge. It projects the sphered data onto a subspace to generate the potential anomaly component.

3. **Sparse Cardinality (SC):** SC is incorporated to reduce the impact of noise on anomaly detection. It selects a subset of the most relevant features or components from the potential anomaly component, thus improving the accuracy of anomaly identification.

4. **RX-AD Implementation:** The refined potential anomaly component undergoes RX-AD (Reed-Xiaoli anomaly detection) for the final detection of anomalies.

The DS-UTSSC method demonstrates competitive performance compared to other existing models, such as LRaSR-based approaches and AE-based methods, in hyperspectral anomaly detection. This method has practical implications in various fields, including remote sensing, surveillance, and target detection, where accurate anomaly detection is crucial for real-world applications.

In summary, the DS-UTSSC method enhances hyperspectral anomaly detection by removing background noise, generating a potential anomaly component through unsupervised target detection, reducing noise impact with sparse cardinality, and implementing a robust RX-AD algorithm.

---

[5]S. Shriram and E. Sivasankar, "Anomaly Detection on Shuttle data using Unsupervised Learning Techniques," 2019 International Conference on Computational Intelligence and Knowledge Economy (ICCIKE), Dubai, United Arab Emirates, 2019, pp. 221-225, doi: 10.1109/ICCIKE47802.2019.9004325.



**Clustering-Based Anomaly Detection Methods**

*Overview:*
The literature on clustering-based anomaly detection methods can be categorized into two main types:

1. **Clustering with Outliers:** These methods aim to partition data into clusters while identifying outliers or anomalies that do not fit into any cluster. The basic idea is to use clustering to define a representation of normal behavior and flag data points that deviate significantly from these clusters.

2. **Density-Based Approaches:** In these methods, the focus is on estimating the data density in feature space. Data points that fall in low-density regions are considered anomalies. The core idea is that anomalies have lower density compared to normal data points.

*Review:*
- *Clustering with Outliers:* Representative methods include k-means clustering with an outlier cluster, DBSCAN with a noise cluster, and GMM-based clustering with a small cluster for anomalies. These methods are effective in separating anomalies from normal data by isolating them in dedicated clusters.

- *Density-Based Approaches:* Methods like LOF (Local Outlier Factor) and DBSCAN (Density-Based Spatial Clustering of Applications with Noise) estimate data density. LOF computes a local density ratio for each data point, while DBSCAN identifies regions of high density as clusters. Anomalies are data points with significantly lower density.

*Comparison:*
- *Clustering with Outliers vs. Density-Based Approaches:* Clustering with outliers explicitly assigns anomalies to distinct clusters, providing clear separation. Density-based approaches rely on estimating data density, which may be more suitable when anomalies have varying densities. However, clustering with outliers may be more intuitive for interpretation.

- *Pros and Cons:* Clustering with outliers provides explicit anomaly clusters but can be sensitive to the choice of clustering algorithm and parameter settings. Density-based approaches are robust to cluster shapes but may struggle with high-dimensional data. Both methods are unsupervised and suitable for scenarios with limited labeled data.

In summary, clustering-based anomaly detection methods offer effective ways to identify anomalies within data. Clustering with outliers and density-based approaches provide different strategies for anomaly detection, each with its own strengths and limitations, making them adaptable to various real-world scenarios.



----




[6] M. P. Handayani, G. Antariksa and J. Lee, "Anomaly Detection in Vessel Sensors Data with Unsupervised Learning Technique," 2021 International Conference on Electronics, Information, and Communication (ICEIC), Jeju, Korea (South), 2021, pp. 1-6, doi: 10.1109/ICEIC51217.2021.9369822.


**Anomaly Detection in Vessel Sensor Data**

*Main Categories of Anomaly Detection Methods:*
- **Isolation Forest:** This technique randomly selects features and uses isolation trees to identify anomalies efficiently. It's suited for high-dimensional datasets and effective in detecting outliers in sensor data.
- **t-SNE (t-Distributed Stochastic Neighbor Embedding):** t-SNE reduces data dimensionality and visualizes complex relationships among data points, aiding in anomaly detection in lower-dimensional space.

*Suitable Scenarios:*
- **Isolation Forest:** Ideal for detecting anomalies in high-dimensional vessel sensor data, capable of handling large datasets and pinpointing outliers.
- **t-SNE:** Useful for dimensionality reduction and visualization of complex data relationships, aiding in anomaly identification.

*Literature Survey:*
- The paper focuses on using unsupervised learning techniques for anomaly detection in vessel sensor data, emphasizing the importance of preventing engine failures and reducing maintenance costs.
- It employs the Isolation Forest algorithm to identify anomalies and utilizes t-SNE for dimensionality reduction.

*Anomaly Detection:*
- Anomaly detection involves identifying patterns or instances significantly deviating from the expected behavior in a dataset.
- Motivation lies in recognizing unusual events indicative of potential issues, threats, or improvement opportunities.
- Examples include detecting fraud in financial transactions, identifying network intrusions, monitoring equipment for faults, and spotting anomalies in vessel sensor data.
- Anomaly detection methods encompass statistical-based, machine learning-based, and hybrid approaches.

*Sensor System in Ships for Engine Anomaly Detection:*
- Ships employ sensor systems to monitor engine status and detect anomalies that may lead to engine failures.
- These systems collect data from various sensors measuring parameters like temperature, pressure, vibration, and fuel consumption.
- Unsupervised learning techniques, including the Isolation Forest algorithm, analyze sensor data to identify anomalies.
- Anomalies may indicate engine abnormalities such as unusual temperature, pressure, vibrations, or fuel consumption, prompting early maintenance recommendations to prevent failures and reduce costs.

*Purpose of Using Isolation Forest:*
- The Isolation Forest algorithm serves to detect anomalies in vessel sensor data.
- It identifies data instances deviating significantly from normal behavior.
- Well-suited for high-dimensional datasets and capable of efficiently handling large data volumes.
- Operates by randomly selecting features and creating isolation trees to isolate anomalies effectively.
- Helps identify potential engine malfunctions or issues by flagging anomalies in the sensor data.

*Reducing Maintenance Costs through Anomaly Detection:*
- Detecting anomalies in sensor data facilitates early identification of engine issues, enabling proactive maintenance.
- Timely maintenance prevents minor issues from escalating into costly failures.
- Prevents unexpected breakdowns, reducing expensive emergency repairs and downtime.
- Identifies trends and patterns in sensor data, addressing root causes of anomalies and improving engine reliability and lifespan.
- Ultimately leads to reduced maintenance costs over the long term.

This summary provides an overview of the key concepts and contributions of the paper related to anomaly detection in vessel sensor data.

---



[7]T. Zoppi, A. Ceccarelli and A. Bondavalli, "Into the Unknown: Unsupervised Machine Learning Algorithms for Anomaly-Based Intrusion Detection," 2020 50th Annual IEEE-IFIP International Conference on Dependable Systems and Networks-Supplemental Volume (DSN-S), Valencia, Spain, 2020, pp. 81-81, doi: 10.1109/DSN-S50200.2020.00044.



**Anomaly-Based Intrusion Detection with Unsupervised Machine Learning Algorithms**

*Main Categories of Anomaly Detection Methods:*
- **Statistical-based methods:** Utilize statistical techniques to model normal system behavior and identify deviations. Suitable for low-dimensional data with well-defined statistical properties.
- **Machine learning-based methods:** Employ machine learning algorithms to learn patterns from data and detect anomalies based on deviations. Suitable for high-dimensional data with undefined statistical properties.
- **Clustering-based methods:** Group similar instances together and identify outliers as anomalies. Effective when anomalies form distinct clusters.
- **Neural network-based methods:** Use neural networks to learn normal behavior and detect deviations. Suitable for high-dimensional data with complex relationships.
- **Graph-based methods:** Model data relationships as graphs and detect anomalies based on unusual patterns or connections.
- **Ensemble methods:** Combine multiple anomaly detection techniques to enhance detection accuracy, suitable for diverse anomaly types.

*Anomaly Detection and Its Goals:*
- Anomaly detection identifies patterns deviating from expected behavior, crucial for detecting intrusions, zero-day attacks, and system failures.
- Examples include spotting unusual network traffic, fraudulent transactions, or abnormal system logs.
- Methods categorized as supervised, unsupervised, or semi-supervised.
- Unsupervised methods, the focus here, classify normal and anomalous behaviors without relying on labeled attack data.

*Machine Learning Algorithms for Anomaly Detection:*
- Unsupervised learning: Trains on normal behavior patterns to flag deviations as anomalies.
- Feature extraction: Captures relevant data features indicating abnormal behavior.
- Clustering techniques: Group similar instances; anomalies are those not belonging to any cluster.
- Statistical methods: Utilize statistical techniques like Gaussian distribution or outlier detection.
- Ensemble methods: Combine multiple models for improved detection.
- Deep learning: Employs deep neural networks like autoencoders or recurrent networks for anomaly detection.
- Online learning: Adapts in real-time to detect evolving anomalies.
- Domain-specific approaches: Customizes algorithms for specific domains, enhancing detection accuracy.

*Types of Security Threats Detectable by Anomaly Detection:*
- Intrusions: Identifies unauthorized access or suspicious activities.
- Zero-day attacks: Detects previously unknown attack patterns.
- Malware and viruses: Flags unusual behavior indicative of infections.
- Insider threats: Identifies abnormal actions by authorized users.
- Network anomalies: Spots unusual network traffic patterns.
- System failures: Detects deviations from normal system behavior.
- Fraud detection: Identifies unusual patterns in transactions or user activities.

*Unsupervised Algorithms in Intrusion Detection:*
- Unsupervised algorithms do not require labeled data for training.
- They excel at identifying unknown anomalies, including zero-day attacks.
- Flexibility to adapt to different domains without retraining.
- Scalable for large datasets without manual labeling.
- Ability to detect novel attack patterns and adapt to evolving threats, enhancing intrusion detection accuracy.

This summary provides an overview of the concepts and advantages associated with using unsupervised machine learning algorithms for anomaly-based intrusion detection.

---



[8]A. B. Nassif, M. A. Talib, Q. Nasir and F. M. Dakalbab, "Machine Learning for Anomaly Detection: A Systematic Review," in IEEE Access, vol. 9, pp. 78658-78700, 2021, doi: 10.1109/ACCESS.2021.3083060.



**Anomaly Detection Methods:**

*Unsupervised Methods:*
- Do not require labeled data.
- Identify anomalies based on deviations from normal patterns.
- Examples include Gaussian Mixture Models (GMM) and k-means clustering.
- Suitable when labeled data is scarce or unavailable.

*Supervised Methods:*
- Rely on labeled data to train a model that can classify anomalies.
- Examples include Support Vector Machines (SVM) and Random Forests.
- Suitable when labeled data is available and anomalies are well-defined.

*Semi-Supervised Methods:*
- Utilize a combination of labeled and unlabeled data for training.
- Learn normal patterns from labeled data and identify deviations in unlabeled data.
- Suitable when limited labeled data is available.

*Deep Learning Methods:*
- Leverage deep neural networks to automatically learn complex patterns and detect anomalies.
- Examples include Autoencoders and Variational Autoencoders.
- Suitable for high-dimensional data with complex patterns.

*Anomaly Detection:*
- Identifies and extracts anomalous components from data.
- Motivated by the need to detect unusual patterns or outliers deviating from expected behavior.
- Applied in various domains, including finance, cybersecurity, healthcare, manufacturing, and environmental monitoring.
- Categories of anomaly detection methods: supervised, unsupervised, and semi-supervised.

**Main Categories of Anomaly Detection Methods (Reiterated):**

- **Supervised Anomaly Detection:** Requires labeled data, suitable for well-defined anomalies.
- **Unsupervised Anomaly Detection:** Detects anomalies without labeled data, suitable for scenarios with unknown or changing anomalies.
- **Semi-Supervised Anomaly Detection:** Combines labeled and unlabeled data, suitable for limited labeled data.
- **Hybrid Anomaly Detection:** Combines multiple techniques for improved detection accuracy.
- **Time Series Anomaly Detection:** Focuses on anomalies in time series data.
- **Network-based Anomaly Detection:** Analyzes network structure and behavior to detect anomalies.

**Perspectives for Analyzing ML Models for Anomaly Detection:**

- **Applications of Anomaly Detection:** Identified 43 different applications in various domains.
- **ML Techniques:** Recognized 29 distinct ML models used for anomaly detection.
- **Performance Metrics:** Mentioned the analysis of performance metrics, but specific details are not provided.
- **Classification of Anomaly Detection Methods:** Highlighted the prevalence of unsupervised anomaly detection among researchers compared to other classification-based methods.

The summary provides an overview of the main categories of anomaly detection methods and perspectives for analyzing machine learning models in the context of anomaly detection.

---



[9] H. Deng and X. Li, "Anomaly Detection via Reverse Distillation from One-Class Embedding," 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), New Orleans, LA, USA, 2022, pp. 9727-9736, doi: 10.1109/CVPR52688.2022.00951.


Certainly, here's a concise summary of the key points from the paper on anomaly detection via reverse distillation:

**Main Categories of Anomaly Detection Methods in the Paper:**

- **Statistical Methods:** Assume normal data follows a specific statistical distribution, identifying anomalies as significant deviations. Suitable for low-dimensional data.
- **Machine Learning Methods:** Use algorithms to learn patterns from normal data and identify anomalies as deviations. Suitable for high-dimensional data.
- **Deep Learning Methods:** Proposes a teacher-student (T-S) model with deep neural networks to learn complex representations and detect anomalies in high-dimensional data.

**Classical Anomaly Detection Methods:**
- Includes one-class support vector machine (OC-SVM) and support vector data description (SVDD).
- Proposes DeepSVDD and PatchSVDD for high-dimensional data.

**Generative Models for Anomaly Detection:**
- Generative models like AutoEncoder (AE) and Generative Adversarial Nets (GAN) used for unsupervised anomaly detection.
- Addresses the issue of deep models successfully reconstructing anomalous regions with memory mechanisms and image masking.

**Comparison with Prior Arts:**
- The proposed method surpasses state-of-the-art performance in unsupervised anomaly detection.
- Compares with various prior arts, including LSA, OCGAN, HRN, DAAD, MKD, GT, GANomaly (GN), Uninformed Student (US), PSVDD, MetaFormer (MF), PaDiM (WResNet50), and CutPaste.

**Novelty of the Proposed Method:**
- Introduces a novel teacher-student (T-S) model with a reverse distillation paradigm.
- The student network aims to restore the teacher's multiscale representations, starting from high-level presentations to low-level features.
- Incorporates a trainable one-class bottleneck embedding (OCBE) module to preserve essential information on normal patterns while disregarding anomaly perturbations.

**Anomaly Detection:**
- Refers to identifying patterns or instances significantly deviating from expected behavior within a dataset.

**Motivation and Examples:**
- Motivated by the need to detect unusual or suspicious activities, events, or patterns indicating fraud, errors, or security breaches.
- Examples include credit card fraud detection, network intrusion detection, manufacturing defect identification, medical anomaly detection, and sensor data anomaly detection.

**Knowledge Distillation Improves Unsupervised Anomaly Detection:**
- Diverse Anomalous Representations: The T-S model enhances the diversity of anomalous representations with a teacher encoder and student decoder.
- Reverse Distillation Paradigm: Student network restores teacher's multiscale representations in a "reverse distillation" paradigm.
- One-Class Bottleneck Embedding (OCBE) Module: OCBE preserves essential normal pattern information while disregarding anomalies.
- Improved Performance: Extensive experimentation demonstrates the proposed approach surpasses state-of-the-art performance.

The summary provides an overview of the paper's main categories of anomaly detection methods, classical approaches, generative models, comparison with prior arts, novelty of the proposed method, and the role of knowledge distillation in improving unsupervised anomaly detection.

---



[10] D. Gudovskiy, S. Ishizaka and K. Kozuka, "CFLOW-AD: Real-Time Unsupervised Anomaly Detection with Localization via Conditional Normalizing Flows," 2022 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), Waikoloa, HI, USA, 2022, pp. 1819-1828, doi: 10.1109/WACV51458.2022.00188.


**Main Categories of Anomaly Detection Methods in the Paper:**

- **Unsupervised anomaly detection with localization:** The paper focuses on methods that perform unsupervised anomaly detection while also providing localization of anomalies within the data. These methods are suitable when labeling is infeasible, and anomaly examples are missing in the training data.

- **Conditional normalizing flow framework:** CFLOW-AD is based on a conditional normalizing flow framework, enabling efficient anomaly detection with localization by modeling the complex distribution of encoded features and estimating their likelihood.

- **Discriminatively pretrained encoder:** CFLOW-AD includes a discriminatively pretrained encoder, trained to discriminate between normal and anomalous samples, enhancing its anomaly detection capability.

- **Multi-scale generative decoders:** CFLOW-AD utilizes multi-scale generative decoders to estimate the likelihood of encoded features and generate samples that match the data distribution. This approach captures both local and global characteristics of encoded features, improving anomaly detection and localization.

**Motivation and Examples:**

- Anomaly detection is motivated by the need to identify unusual or suspicious events, especially when labeling is impractical, or anomaly examples are missing in the training data.
- Practical applications include fraud detection, network intrusion detection, manufacturing quality control, and medical diagnosis.

**How CFLOW-AD Achieves Real-Time Processing for Anomaly Detection:**

- CFLOW-AD achieves real-time processing by proposing a computationally and memory-efficient model based on the conditional normalizing flow framework.
- It includes a discriminatively pretrained encoder and multi-scale generative decoders.
- The model is 10x faster and smaller compared to previous state-of-the-art models with the same input setting.
- This efficiency allows CFLOW-AD to process data in real-time while maintaining high accuracy in anomaly detection and localization.

**Advantages of CFLOW-AD over Prior State-of-the-Art Models:**

- CFLOW-AD offers a significant advantage in terms of computational and memory efficiency, making it suitable for real-time processing.
- It achieves high accuracy metrics while maintaining real-time processing capabilities.
- CFLOW-AD's conditional normalizing flow framework enables efficient data processing and likelihood estimation of encoded features.
- Experimental results on the MVTec dataset demonstrate its effectiveness in both anomaly detection and localization tasks.

**How CFLOW-AD Estimates Likelihood of Encoded Features:**

- CFLOW-AD estimates the likelihood of encoded features using its multi-scale generative decoders.
- These decoders explicitly estimate the likelihood of encoded features, enabling the assessment of anomaly scores for input samples.
- The conditional normalizing flow framework allows CFLOW-AD to model the complex distribution of encoded features and generate samples that match the original data distribution.
- Likelihood estimation is performed at multiple scales, capturing both local and global characteristics of encoded features.
- This approach enhances CFLOW-AD's ability to accurately detect and localize anomalies in the input data.

The summary provides an overview of CFLOW-AD's key contributions, its architecture, motivation for anomaly detection, and its efficiency in real-time processing and likelihood estimation of encoded features for anomaly detection with localization.


1. **Literature Review (25%)**
    - Overview of Clustering-Based Anomaly Detection Methods
    - Review of Representative Methods/Papers
    - Comparison of Methods (Pros, Cons, Similarities, Differences)

2. **Algorithm Description (20%)**
    - Definition and Rationale Behind Anomaly Score
    - Pseudo Code and Textual Explanation


Anomaly Detection Schemes 
General steps
Build a profile of the “normal” behavior
Profile can be patterns or summary statistics for the overall population
Use the “normal” profile to detect anomalies
Anomalies are observations whose characteristics differ significantly from the normal profile

---


### Anomaly Score Definition

#### Definition
The anomaly score `A(x)` for a given data point `x` is defined as the weighted sum of its Euclidean distance to its nearest cluster center and its distance rank within its cluster. Mathematically,

\[
A(x) = w_1 \times D(x, C_i) + w_2 \times R(x, C_i)
\]

- \(D(x, C_i)\) is the Euclidean distance between point \(x\) and its nearest cluster center \(C_i\).
- \(R(x, C_i)\) is the rank of \(x\) when all points in cluster \(C_i\) are sorted by their distance to \(C_i\).
- \(w_1\) and \(w_2\) are weights such that \(w_1 + w_2 = 1\).

#### Rationale

1. **Euclidean Distance (\(D\))**: Points far from their cluster centers are more likely to be anomalies. 
2. **Distance Rank (\(R\))**: It adds context by considering how a point's distance compares with other points in the same cluster.

#### Example

Suppose we have a cluster with points at distances `[1, 2, 2.5, 3, 10]` from the cluster center. If \(w_1 = 0.6\) and \(w_2 = 0.4\),

- A(x) for \(x\) at distance 10 would be \(0.6 \times 10 + 0.4 \times 5 = 8\).

### Algorithm Design

#### Pseudo-Code
```plaintext
1. Initialize K, w1, w2, and Anomaly_Threshold
2. Standardize the dataset
3. Train K-Means model on standardized dataset to get cluster centers C_1, C_2, ..., C_K
4. Initialize Anomaly_Scores as empty list
5. For each data point x in dataset:
    a. Find nearest cluster center C_i
    b. Calculate D(x, C_i) 
    c. Calculate R(x, C_i) within its cluster
    d. Calculate Anomaly_Score = w1 * D(x, C_i) + w2 * R(x, C_i)
    e. Append Anomaly_Score to Anomaly_Scores
6. For each Anomaly_Score in Anomaly_Scores:
    a. If Anomaly_Score > Anomaly_Threshold:
        Label corresponding data point as anomaly
```

#### Description of Steps

1. **Initialize Parameters**: Set the number of clusters \(K\), weights \(w_1\) and \(w_2\), and a threshold for flagging anomalies.
   
2. **Standardize Data**: Center and scale the data to have zero mean and unit variance. This is crucial for distance-based algorithms like K-means.

3. **Train K-Means**: Cluster the standardized dataset using K-means to obtain cluster centers.

4. **Initialize Anomaly Scores**: Create an empty list to store the anomaly scores.

5. **Calculate Anomaly Scores**: 
    - **Find Nearest Cluster**: For each point, find the nearest cluster center.
    - **Distance and Rank**: Calculate the Euclidean distance and distance rank for each point within its cluster.
    - **Compute Score**: Compute the anomaly score as the weighted sum of these distances and ranks.

6. **Label Anomalies**: Loop through the computed anomaly scores. If a score crosses the predefined threshold, label the corresponding data point as an anomaly.

#### Example

Suppose \(K = 3, w_1 = 0.6, w_2 = 0.4, \text{and Anomaly\_Threshold} = 7\). After running K-means, let's consider a data point \(x\) belonging to cluster \(C_1\). Assume \(D(x, C_1) = 5\) and \(R(x, C_1) = 3\).

- \( \text{Anomaly\_Score} = 0.6 \times 5 + 0.4 \times 3 = 4.2\)
- Since \(4.2 < 7\), \(x\) is not labeled as an anomaly.

This algorithm efficiently uses both distance and rank metrics, weighted appropriately, to identify anomalies. The use of a threshold allows for fine-tuning of the sensitivity of the anomaly detection process.

---


### Parametric vs Non-Parametric Methods

The K-means based approach is a non-parametric method because it does not make any assumptions about the underlying distribution of the data. It solely relies on the structure learned from the data.

### Parametric Anomaly Detection using Gaussian Mixture Model (GMM)

GMM is a probabilistic model that assumes that the data is generated from a mixture of several Gaussian distributions. The parameters of these distributions can be estimated from the data.

#### Anomaly Score Definition

The anomaly score `A(x)` for a data point `x` is defined as the negative log-likelihood of that point given the learned Gaussian Mixture Model:

\[
A(x) = -\log \left( \sum_{i=1}^{K} w_i \times \mathcal{N}(x; \mu_i, \Sigma_i) \right)
\]

- \( w_i \) is the weight of the \(i\)-th Gaussian component.
- \( \mathcal{N}(x; \mu_i, \Sigma_i) \) is the Gaussian distribution with mean \( \mu_i \) and covariance matrix \( \Sigma_i \).

#### Rationale

The lower the likelihood of observing a given data point under the fitted model, the higher its anomaly score. Thus, this score captures the anomalousness of data objects effectively.

#### Algorithm Design

##### Pseudo-Code

```plaintext
1. Initialize K, Anomaly_Threshold
2. Standardize the dataset
3. Fit a GMM model to the dataset, learn {w_i, mu_i, Sigma_i} for i = 1,...,K
4. Initialize Anomaly_Scores as empty list
5. For each data point x in dataset:
    a. Calculate A(x) using -log likelihood
    b. Append A(x) to Anomaly_Scores
6. For each Anomaly_Score in Anomaly_Scores:
    a. If Anomaly_Score > Anomaly_Threshold:
        Label corresponding data point as anomaly
```

##### Description of Steps

1. **Initialize Parameters**: Set \(K\) and the anomaly threshold.
2. **Standardize Data**: Standardization is essential for any distance-based model.
3. **Fit GMM**: Fit a Gaussian Mixture Model to the data.
4. **Initialize Anomaly Scores**: An empty list to store the calculated anomaly scores.
5. **Calculate Anomaly Scores**: Use the negative log-likelihood formula to calculate the anomaly score for each data point.
6. **Label Anomalies**: Any data point with an anomaly score greater than the threshold is labeled as an anomaly.

#### Example

Assume \(K = 3\) and \( \text{Anomaly\_Threshold} = 20\). After fitting the GMM, let's say we want to calculate the anomaly score for a data point \(x\).

- Assume \( w_1 = 0.5, \mu_1 = 0, \Sigma_1 = 1 \)
- Assume \( w_2 = 0.3, \mu_2 = 3, \Sigma_2 = 1 \)
- Assume \( w_3 = 0.2, \mu_3 = -3, \Sigma_3 = 1 \)

The anomaly score \( A(x) \) would be calculated using the negative log-likelihood formula.

- If \( A(x) > 20 \), \(x\) is labeled as an anomaly.

The parametric GMM-based method allows us to explicitly model the underlying distributions, making the algorithm's assumptions clear and enabling a probabilistic interpretation of anomalies.

---

### K-means Based Anomaly Detection Algorithm

#### Definition of the Anomaly Score

The anomaly score `A(x)` for a data point `x` is defined as the normalized distance from the point to its nearest cluster center multiplied by the cluster's standard deviation. The formula is:

\[
A(x) = \frac{D(x, C_i)}{\max(D)} \times \sigma(C_i)
\]

- \(D(x, C_i)\) is the Euclidean distance between point \(x\) and its nearest cluster center \(C_i\).
- \(\max(D)\) is the maximum distance any point has to its nearest cluster center across all clusters.
- \(\sigma(C_i)\) is the standard deviation of distances of all points in cluster \(C_i\) to \(C_i\).

#### Rationale

- **Normalized Distance**: Normalizing the distance by the maximum distance across all clusters puts all points on a comparable scale.
  
- **Cluster Variance (\(\sigma\))**: Multiplying by the standard deviation of the distances in the cluster accounts for the cluster's density. Points in sparse clusters are more likely to be anomalous.

#### Example

Suppose a point \(x\) has a distance of 10 units from its nearest cluster center \(C_1\), the maximum distance across all clusters is 20, and the standard deviation of distances in \(C_1\) is 2. Then, \( A(x) = \frac{10}{20} \times 2 = 1 \).

#### Pseudo-Code

```plaintext
1. Initialize K, Anomaly_Threshold
2. Standardize the dataset
3. Train K-means model to get cluster centers C_1, ..., C_K
4. Calculate max(D) across all clusters
5. Initialize Anomaly_Scores = []
6. For each point x in dataset:
    a. Find nearest cluster center C_i
    b. Calculate D(x, C_i)
    c. Calculate sigma(C_i)
    d. A(x) = (D(x, C_i) / max(D)) * sigma(C_i)
    e. Append A(x) to Anomaly_Scores
7. For each A(x) in Anomaly_Scores:
    a. If A(x) > Anomaly_Threshold:
        Label x as anomaly
```

#### Description of Steps

1. **Initialize Parameters**: Choose \(K\) and set the anomaly threshold.
2. **Standardize Data**: Center and scale the data.
3. **Train K-means**: Obtain cluster centers.
4. **Calculate Max Distance**: Find the maximum distance any point has to its nearest cluster center.
5. **Initialize Anomaly Scores**: Create an empty list.
6. **Calculate Anomaly Scores**: For each data point, calculate \(A(x)\) as defined.
7. **Label Anomalies**: Label points as anomalies if their \(A(x)\) exceeds the threshold.




8. **Application to Dataset (30%)**
    - Algorithm Execution Steps
    - Detected Anomalies and Discussion
    - Data Pre-processing Steps (if needed)

### 5-Day Plan

**Day 1: Research**
- Search and review literature focusing on clustering-based anomaly detection methods. Aim to finalize 8-10 related papers for an in-depth review.
  
**Day 2: Introduction & Literature Review Writing**
- Write the Introduction.
- Write the Literature Review based on the selected papers.

**Day 3: Algorithm Design**
- Sketch out the anomaly score, considering factors from k-means that will help in detecting anomalies.
- Formulate the algorithm in pseudo code and provide a text-based explanation.

**Day 4: Application and Experimentation**
- Use SAS Enterprise Miner for k-means clustering on `a2-housing.csv`.
- Apply other steps of the algorithm using tools at your disposal, identify anomalies, and evaluate results.

**Day 5: Finalization and Review**
- Write the section on applying the algorithm to the dataset.
- Review the entire paper, ensuring that it meets all the specified requirements and is error-free. 

