# TypeNet Benchmark 
TypeNet Series Benchmark for development of authentication keystroke technologyes based on deep neuronal networks.

## INSTRUCTIONS FOR DOWNLOADING TypeNet Benchmark
1) [Download license agreement](http://atvs.ii.uam.es/atvs/licenses/BeCAPTCHA-Mouse_License_Agreement.pdf), send by email one signed and scanned copy to **atvs@uam.es** according to the instructions given in point 2.
 
 
2) Send an email to **atvs@uam.es**, as follows:

   *Subject:* **[DATABASE benchmark: TypeNet Benchmark]**

   Body: Your name, e-mail, telephone number, organization, postal mail, purpose for which you will use the database, time and date at which you sent the email with the signed license agreement.
 

3) Once the email copy of the license agreement has been received at ATVS, you will receive an email with a username, a password, and a time slot to download the database.
 

4) [Download the benchmark](http://atvs.ii.uam.es/atvs/intranet/free_DB/beCAPTCHA), for which you will need to provide the authentication information given in step 4. After you finish the download, please notify by email to **atvs@uam.es** that you have successfully completed the transaction.
 

5) For more information, please contact: **atvs@uam.es**


## DESCRIPTION OF TypeNet benchmark
This benchmark contains the embedding vectors from 130K subjects generated during free-text typing in both touchscreen virtual (30K subjects) and physical keyboards (100K subjects) scenarios. These embedding vectors are calculated with TypeNet, a recurrent deep neuronal network aimed to recognize individuals at large scale based on their typing behaviours. Aditionally, we provide a experimental protocol to reproduce the authentication results obtained with TypeNet in Acien *et al.* [1] paper.


**Keystroke Datasets**  
The embedding vectors are obtained when passing through TypeNet networks the keystroke sequences acquired from the two Aalto University Datasets: 1) Dhakal *et al.* [2] dataset, which comprises more than 5GB of keystroke data collected in desktop keyboards from 168K participants; and 2) Palin  *et al.* [3] dataset, which comprises almost 4GB of keystroke data collected in mobile devices from 260K participants. The data were collected following the same procedure for both datasets. The acquisition task required subjects to memorize English sentences and then type them as quickly and accurate as they could. The English sentences were selected randomly from a set of 1525 examples taken from the Enron mobile email and Gigaword Newswire corpus. The example sentences contained a minimum of 3 words and a maximum of 70 characters. Note that the sentences typed by the participants could contain more than 70 characters because each participant could forget or add new characters when typing. All participants in the Dhakal database completed 15 sessions (i.e. one sentence for each session) on either a desktop or a laptop physical keyboard. However, in the Palin dataset the participants who finished at least 15 sessions are only 23% (60K participants) over 260 participants that started the typing test.


**TypeNet Architecture**  
The TypeNet architecture is depicted in Fig. 1. It is composed of two Long Short-Term Memory (LSTM) layers of 128 units (*tanh* activation function). Between the LSTM layers, we perform batch normalization and dropout at a rate of 0.5 to avoid overfitting. Additionally, each LSTM layer has a recurrent dropout rate of 0.2. 

In order to train TypeNet with sequences of different lengths *N* within a single batch, we truncate the end of the input sequence when *N>M* and zero pad at the end when *N<M*, in both cases to the fixed size *M*. The embedding vector provided are obtained for keystroke sequence of size *M* = 50 keys.
Finally, the output of the model **f(x)** is an array of size 1X128 that represents the embedding feature vectors that we will employ to authenticate subjects.

![](https://github.com/BiDAlab/TypeNet/blob/main/TypeNet_architecture.png)
**Figure 1. Architecture of TypeNet for free-text keystroke sequences. The input x is a keystroke sequence of size *M*=50 keys and the output f(x) is an embedding vector with shape 1X128.**

As depicted in Fig .2, TypeNet is trained with three loss functions (softmax, contrastive and triplet loss), and therefore, three different TypeNet versions (i.e. one for each loss function) are employed to calculated the embedding vectors for both scenarios: desktop scenario and mobile scenario, with the models trained with Dhakal and Palin databases, respectively. For the desktop scenario, we train the models using only the first 68K subjects from the Dhakal dataset. For the Softmax function we train a model with *C* = 10K subjects (we could not train with more subjects due to hardware limitations) which means 15 X 10K = 150K training keystroke sequences (the remaining 58K subjects were discarded). For the Contrastive loss we generate genuine and impostor pairs using all the 15 keystroke sequences available for each subject. This provides us with 15 X 67,999 X 15 = 15.3 millions of impostor pair combinations and 15 X 14/2 = 105 genuine pair combinations for each subject. The pairs were chosen randomly in each training batch ensuring that the number of genuine and impostor pairs remains balanced (512 pairs in total in each batch including impostor and genuine pairs). Similarly, we randomly chose triplets for the Triplet loss training.

![](https://github.com/BiDAlab/TypeNet/blob/main/training3.png)
**Figure 2. Learning architecture of TypeNet for the different loss functions a) Softmax loss, b) Contrastive loss, and c) Triplet loss. The goal is to find the most discriminant embedding space f(x).**

The remaining 100K subjects were employed only to test the desktop models, so there is no data overlap between the two groups of subjects (open-set authentication paradigm). The same protocol was employed for the mobile scenario but adjusting the amount of subjects employed to train and test. In order to have balanced subsets close to the desktop scenario, we divided by half the Palin database. It means that 30K subjects were employed to train the models, generating 15 X 29,999 X 15 = 6.75 millions of impostor pair combinations and 15 X 14/2 = 105 genuine pair combinations for each subject, meanwhile the other 30K subjects were employed to test the mobile TypeNet models. Once again 10K subjects were employed to train the models based on the Softmax function.

The embedding feature vectors provided in this repository come from these 100K test subjects for the desktop scenario and the 30K test subjects for the mobile scenario.


**Experimental Protocol**
We authenticate subjects by comparing gallery samples **x<sub>i,g</sub>** belonging to the subject *i* in the test set to a query sample  **x<sub>j,q</sub>** from either the same subject (genuine match *i = j*) or another subject (impostor match *i ≠ j*). The test score is computed by averaging the Euclidean distances between each gallery embedding vector **f(x<sub>i,g</sub>)** and the query embedding vector **f(x<sub>i,q</sub>)**  as follows:
\begin{equation}
\label{score}
     \textit{s}_{i,j}^q= \frac{1}{G}\sum_{g=1}^{G} ||\textbf{f}(\textbf{x}_{i,g})-\textbf{f}(\textbf{x}_{j,q})||
\end{equation}
where $G$ is the number of sequences in the gallery (i.e. the number of enrollment samples) and $q$ is the query sample of subject $j$. Taking into account that each subject has a total of $15$ sequences, we retain $5$ sequences per subject as test set (i.e. each subject has $5$ genuine test scores) and let $G$ vary between $1 \leq G \leq 10$ in order to evaluate the performance as a function of the number of enrollment sequences.

To generate impostor scores, for each enrolled subject we choose one test sample from each remaining subject. We define $k$ as the number of enrolled subjects. In our experiments, we vary $k$ in the range $100 \leq k \leq K$, where $K = 100$,$000$ for the desktop TypeNet models and $K = 30$,$000$ for the mobile ones. Therefore each subject has $5$ genuine scores and $k-1$ impostor scores. Note that we have more impostor scores than genuine ones, a common scenario in keystroke dynamics authentication. The results reported in the next section are computed in terms of Equal Error Rate (EER), which is the value where False Acceptance Rate (FAR, proportion of impostors classified as genuine) and False Rejection Rate (FRR, proportion of genuine subjects classified as impostors) are equal. The error rates are calculated for each subject and then averaged over all $k$ subjects \cite{2014_IWSB_Aythami_Keystroking}.

#### FILES FORMAT
+ .txt files: it just contains two columns with the **{x̂, ŷ}** mouse coordinates.
  + COLUMN 1: represents the **x̂** coordinate.

  + COLUMN 2: represents the **ŷ** coordinate.

+ .ana files: each row contains a log-normal signal extracted from the synthetic mouse trayectory, this log-normal signal is definded by 6 parameters. One parameter in each column:  

  + COLUMN 1: represents the *D* parameter.

  + COLUMN 2: represents the *t<sub>0</sub>* parameter.

  + COLUMN 3: represents the *μ* parameter.

  + COLUMN 4: represents the *σ* parameter.

  + COLUMNS 5 : represents the *θ<sub>s</sub>* parameter.
  
  + COLUMNS 6 : represents the *θ<sub>e</sub>* parameter.
  
  + COLUMNS 7, 8: are zeros.
  

#### FILES NOMENCLATURE
The nomenclature followed to name the files of the function-based method is: NNNN_y=A_vp=B_task=C.txt

+ NNNN: indicates the number of the sample.

+ A: indicates the shape of the trajectory:

  + 0 = linear.
  
  + 1 = quadratic.
  
  + 2 = exponential.
  
+ B: indicates the velocity profile:

  + 0 = constant velocity.
  
  + 1 = logarithmic velocity.
  
  + 2 = Gaussian velocity.
  
+ C: indicates the task (1-8) of the human mouse database in which the trayectory was synthetized. This is necessary because the function-based method needs the initial [*x̂<sub>1</sub>, ŷ<sub>1</sub>*] and the end [*x̂<sub>M</sub>, ŷ<sub>M</sub>*] points of the human trayectory to synthetyse.


#### REFERENCES
For further information on the benchmark and on different applications where it has been used, we refer the reader to (all these articles are publicly available in the [publications](http://atvs.ii.uam.es/atvs/listpublications.do) section of the BiDA group webpage).

+ [1] A. Acien, A. Morales, J. Fierrez, R. Vera-Rodriguez. BeCAPTCHA-Mouse: Synthetic Mouse Trajectories and Improved Bot Detection. *arXiv:2005.00890*, 2020. [[pdf](https://arxiv.org/pdf/2005.00890.pdf)]

+ [2] C. Shen, Z. Cai, X. Guan, R. Maxion. Performance evaluation of anomalydetection algorithms for mouse dynamics, *Computers & Security*, 45: 156–171, 2014.

+ [3] M. Djioua, R. Plamondon. A new algorithm and system for the characterization of handwriting strokes with delta-lognormal parameters, *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 31(11): 2060–2072, 2009.

Please remember to reference article [1] on any work made public, whatever the form, based directly or indirectly on any part of the BeCAPTCHA-Mouse benchmark.
