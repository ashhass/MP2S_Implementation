# MP2S (Modernizing Public Security Systems) Implementation 

**Project Description**

This project aims to use available technologies for ensuring public security. It presents a possible solution to the recurring problem of public security infringement, 
biased police investigations and inefficient security systems. By building on top of existing police department databases, we automate the process of identifying, reporting and notifying concerned bodies of incidents.

***

**Model Description**

To identify anomalies or incidents on surveillance videos, we use a simple convolutional autoencoder stacked with lstm layers. Despite recent advancements in anomaly detection
models, the methods developed are much more sophisticated with marginal performance improvements. Hence, we opted to use simple but effective architectures that are efficient 
for real-time analysis. 

***

**Evaluation**

We evaluated model performance on various metrics with False Negative Rate being the most important one. The reason is that for this particular application where every unreported incident
has detrimental consequences, the goal is to reduce the number of unreported incidents. Here is a table with results:

TPR/Recall | FPR | FNR/Miss Rate | Precision 
 :-- | :--: | :--: | :--: 
0.923 | 0.74 | 0.076 | 0.674 

***

**How To Use**

Although we specifically built this system for the police department usecase, it can be applied to various other departments such as firefighting, ambulances etc. 
It can be easily used when connected to existing database and surveillance systems. We intentionally used very clear interfaces to reduce any possible confusions and 
facilitate ease of use. 

***

**Future Work**

Our model can be continually improved after deployment. We make this possible by saving the time frames wherein the model predicts presence of anomaly. Through our system,
officers are prompted to enter details after every incident. This information can later be used to further train the model on newly acquired data. We believe
this has the potential to significantly improve performance and ensures continuous relevance of our system. 
