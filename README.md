# TypeNet Benchmark 
TypeNet Benchmark for development of authentication keystroke technologies based on deep neuronal networks.

## INSTRUCTIONS FOR DOWNLOADING TypeNet Benchmark

1) [Download license agreement](http://atvs.ii.uam.es/atvs/licenses/TypeNet_License_Agreement.pdf), send by email one signed and scanned copy to **atvs@uam.es** according to the instructions given in point 2.
 
 
2) Send an email to **atvs@uam.es**, as follows:

   *Subject:* **[DATABASE benchmark: TypeNet Benchmark]**

   Body: Your name, e-mail, telephone number, organization, postal mail, purpose for which you will use the database, time and date at which you sent the email with the signed license agreement.
 

3) Once the email copy of the license agreement has been received at ATVS, you will receive an email with a username, a password, and a time slot to download the database.
 

4) [Download the benchmark](http://atvs.ii.uam.es/atvs/intranet/free_DB/TypeNet), for which you will need to provide the authentication information given in step 4. After you finish the download, please notify by email to **atvs@uam.es** that you have successfully completed the transaction.
 

5) For more information, please contact: **atvs@uam.es**




## DESCRIPTION OF TypeNet Benchmark
This benchmark contains the embedding vectors from 130K subjects generated during free-text typing in both touchscreen virtual (30K subjects) and physical keyboards (100K subjects) scenarios. These embedding vectors are calculated with TypeNet, a recurrent neuronal network aimed to recognize individuals at large scale based on their typing behaviours. Aditionally, we provide an experimental protocol to reproduce the authentication results obtained with TypeNet in Acien *et al.* [1] paper.


**Keystroke Datasets**  
The embedding vectors are obtained when passing through TypeNet networks the keystroke sequences acquired from the two Aalto University Datasets: 1) Dhakal *et al.* [2] dataset, which comprises more than 5GB of keystroke data collected in desktop keyboards from 168K participants; and 2) Palin  *et al.* [3] dataset, which comprises almost 4GB of keystroke data collected in mobile devices from 260K participants. The data were collected following the same procedure for both datasets. The acquisition task required subjects to memorize English sentences and then type them as quickly and accurate as they could. The English sentences were selected randomly from a set of 1525 examples taken from the Enron mobile email and Gigaword Newswire corpus. The example sentences contained a minimum of 3 words and a maximum of 70 characters. Note that the sentences typed by the participants could contain more than 70 characters because each participant could forget or add new characters when typing. All participants in the Dhakal database completed 15 sessions (i.e. one sentence for each session) on either a desktop or a laptop physical keyboard. However, in the Palin dataset the participants who finished at least 15 sessions are only 23% (60K participants) over 260 participants that started the typing test.


**TypeNet Architecture**  
The TypeNet architecture is depicted in Fig. 1. It is composed of two Long Short-Term Memory (LSTM) layers of 128 units (*tanh* activation function). Between the LSTM layers, we perform batch normalization and dropout at a rate of 0.5 to avoid overfitting. Additionally, each LSTM layer has a recurrent dropout rate of 0.2. 

In order to train TypeNet with sequences of different lengths *N* within a single batch, we truncate the end of the input sequence when *N>M* and zero pad at the end when *N<M*, in both cases to the fixed size *M*. The embedding vector provided are obtained for keystroke sequence of size *M* = 50 keys.
Finally, the output of the model **f(x)** is an array of size 1X128 that represents the embedding feature vectors that we will employ to authenticate subjects.

![](https://github.com/BiDAlab/TypeNet/blob/main/TypeNet_architecture.png)
**Figure 1. Architecture of TypeNet for free-text keystroke sequences. The input x is a keystroke sequence of size *M*=50 keys and the output f(x) is an embedding vector with shape 1X128.**

As depicted in Fig .2, TypeNet is trained with three loss functions (softmax, contrastive and triplet loss), and therefore, trhee different TypeNet versions (i.e. one for each loss function) are employed to calculated the embedding vectors for both scenarios: desktop scenario and mobile scenario, with the models trained with Dhakal and Palin databases, respectively. For the desktop scenario, we train the models using only the first 68K subjects from the Dhakal dataset. For the Softmax function we train a model with *C* = 10K subjects (we could not train with more subjects due to hardware limitations) which means 15 X 10K = 150K training keystroke sequences (the remaining 58K subjects were discarded). For the Contrastive loss we generate genuine and impostor pairs using all the 15 keystroke sequences available for each subject. This provides us with 15 X 67,999 X 15 = 15.3 millions of impostor pair combinations and 15 X 14/2 = 105 genuine pair combinations for each subject. The pairs were chosen randomly in each training batch ensuring that the number of genuine and impostor pairs remains balanced (512 pairs in total in each batch including impostor and genuine pairs). Similarly, we randomly chose triplets for the Triplet loss training.

![](https://github.com/BiDAlab/TypeNet/blob/main/training3.png)
**Figure 2. Learning architecture of TypeNet for the different loss functions a) Softmax loss, b) Contrastive loss, and c) Triplet loss.**

The remaining 100K subjects were employed only to test the desktop models, so there is no data overlap between the two groups of subjects (open-set authentication paradigm). The same protocol was employed for the mobile scenario but adjusting the amount of subjects employed to train and test. In order to have balanced subsets close to the desktop scenario, we divided by half the Palin database. It means that 30K subjects were employed to train the models, generating 15 X 29,999 X 15 = 6.75 millions of impostor pair combinations and 15 X 14/2 = 105 genuine pair combinations for each subject, meanwhile the other 30K subjects were employed to test the mobile TypeNet models. Once again 10K subjects were employed to train the models based on the Softmax function.

The embedding feature vectors provided in this repository come from these 100K test subjects for the desktop scenario and the 30K test subjects for the mobile scenario.

**Experimental Protocol**  
We authenticate subjects by comparing gallery samples **x<sub>i,g</sub>** belonging to the subject *i* in the test set to a query sample  **x<sub>j,q</sub>** from either the same subject (genuine match *i = j*) or another subject (impostor match *i ≠ j*). The test score is computed by averaging the Euclidean distances between each gallery embedding vector **f(x<sub>i,g</sub>)** and the query embedding vector **f(x<sub>j,q</sub>)**  as follows:
<img src="https://github.com/BiDAlab/TypeNet/blob/main/equation.png">

where *G* is the number of sequences in the gallery (i.e. the number of enrollment samples) and *q* is the query sample of subject *j*. Taking into account that each subject has a total of 15 sequences, we retain 5 sequences per subject as test set (i.e. each subject has 5 genuine test scores) and let *G* vary between 1 ≤ *G* ≤ 10 in order to evaluate the performance as a function of the number of enrollment sequences.

To generate impostor scores, for each enrolled subject we choose one test sample from each remaining subject. We define *k* as the number of enrolled subjects. In our experiments, we vary *k* in the range 100 ≤ *k* ≤ *K*, where *K* = 100,000 for the desktop TypeNet models and *K* = 30,000 for the mobile ones. Therefore each subject has 5 genuine scores and *k*-1 impostor scores.

#### FILES FORMAT
+ .npy files: a matrix that contains the embedding vectors of dimensions (Subject, Session, Embedding vector).

  + Subject index: from 0 to 100,000 for desktop and from 0 to 30,000 for mobile.

  + Session index : from 0 to 15.
  
  + Embedding vector: a vector of size 1 X 128.
  

#### FILES NOMENCLATURE
The nomenclature followed to name the .npy files  is: *Embedding_vectors_LOSS_SCENARIO.npy*

+ LOSS: indicates the loss function employed to train the TypeNet model and calculate the embedding vectors.

  + Softmax = softmax loss function (*categorical_crossentropy loss with softmax activation function*).
  
  + Contrastive = contrastive loss function.
  
  + Triplet = triplet loss function.
  

+ SCENARIO: indicates whether the embedding vectors are extracted from the desktop or mobile dataset.

  + Mobile = mobile dataset.
  
  + Desktop = desktop dataset.
  
  
#### EXAMPLE USAGE
We provide an example of the experimental protocol bellow:
```python

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

#Function to calculate the accuracy as the inverse of Equal Error Rate
def eer_compute(scores_g, scores_i): 

    far = []
    frr = []
    ini=min(np.concatenate((scores_g, scores_i)))
    fin=max(np.concatenate((scores_g, scores_i)))
    
    paso=(fin-ini)/10000
    threshold = ini-paso
    while threshold < fin+paso:
        far.append(len(np.where(scores_i >= threshold)[0])/len(scores_i))
        frr.append(len(np.where(scores_g < threshold)[0])/len(scores_g))
        threshold = threshold + paso
    
    gap = abs(np.asarray(far) - np.asarray(frr))
    j = np.where(gap==min(gap))[0]
    index = j[0]
    return ((far[index]+frr[index])/2)*100, frr,far


#################################
##             Main            ## 
################################# 
# Load the embedding vectors
Matrix_embbeding = np.load('Embedding_vectors_Triplet_Desktop.npy')

#Number of test users 'k' (K= 100000 in dekstop and K= 30000 in mobile) 
NUM_TEST_USERS = 1000 

#The experimental protocol for authentication with different values of 'G'
GALLERY_VALUES = [1,2,5,7,10] #Values of 'G'

for iG in GALLERY_VALUES:              
    NUM_SAMPLES_GALLERY = iG #Number of gallery samples employed ('G')          
    Mean_acc_per_user = []
    
    for genuine_user in range(NUM_TEST_USERS):
                Gallery_matrix = Matrix_embbeding[genuine_user, :NUM_SAMPLES_GALLERY,:] # Gallery matrix
                genuine_matrix = Matrix_embbeding[genuine_user, 10:,:]# Query Genuine matrix: the last 5 sessions of the genuine user
                Y_pos_vec = np.mean(euclidean_distances(Gallery_matrix, genuine_matrix), axis = 0) #Genuine scores
                Impostors_users = np.arange(NUM_TEST_USERS)
                Impostors_users = np.delete(Impostors_users, genuine_user)
                Unknown_matrix = Matrix_embbeding[Impostors_users, 11,:]# Query Unknown matrix: one session for each impostor user
                Y_neg_vec = np.mean(euclidean_distances(Gallery_matrix, Unknown_matrix), axis = 0) #Impostor scores   
                
                ACC,_,_ = eer_compute(Y_pos_vec, Y_neg_vec)
                Mean_acc_per_user.append(ACC)
    
            
    print('Number of genuine sessions employed as gallery: '+str(iG))
    print('Mean EER per user: '+ str(100-np.mean(Mean_acc_per_user)))
```


#### REFERENCES
For further information on the benchmark and on different applications where it has been used, we refer the reader to (all these articles are publicly available in the [publications](http://atvs.ii.uam.es/atvs/listpublications.do) section of the BiDA group webpage).

+ [1] A. Acien, A. Morales, J.V. Monaco, R. Vera-Rodriguez, J. Fierrez, "TypeNet: Deep Learning Keystroke Biometrics," in *IEEE Transactions on Biometrics, Behavior, and Identity Science (TBIOM)*, vol. 4, pp. 57 - 70, 2022. [[pdf](https://arxiv.org/pdf/2101.05570.pdf)]


+ [2] V. Dhakal, A. M. Feit, P. O. Kristensson, and A. Oulasvirta. Observations on typing from 136 million keystrokes, in *Proc. of the ACM CHI Conference on Human Factors in Computing Systems*, 2018.

+ [3] K. Palin, A. M. Feit, S. Kim, P. O. Kristensson, and A.  Oulasvirta. “How do people type on mobile devices? observations from a study with 37,000 volunteers.”  in *Proc.  of  21st  ACM  International  Conferenceon  Human-Computer  Interaction  with  Mobile  Devices  and  Services (MobileHCI’19)*, 2019.

+ [4] A. Acien, J.V. Monaco, A. Morales, R. Vera-Rodriguez, J. Fierrez, “TypeNet: Scaling up Keystroke Biometrics,” in *Proc. of IAPR/IEEE International Joint Conference on Biometrics (IJCB)*, Houston, USA, 2020. [[pdf](https://arxiv.org/pdf/2005.00890.pdf)]

Please remember to reference article [1,4] on any work made public, whatever the form, based directly or indirectly on any part of the TypeNet Benchmark.
