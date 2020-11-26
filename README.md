# TypeNet Benchmark 
TypeNet Benchmark for development of authentication keystroke technologyes based on deep neuronal networks.

## INSTRUCTIONS FOR DOWNLOADING TypeNet Benchmark
1) [Download license agreement](http://atvs.ii.uam.es/atvs/licenses/BeCAPTCHA-Mouse_License_Agreement.pdf), send by email one signed and scanned copy to **atvs@uam.es** according to the instructions given in point 2.
 
 
2) Send an email to **atvs@uam.es**, as follows:

   *Subject:* **[DATABASE benchmark: TypeNet_Benchmark]**

   Body: Your name, e-mail, telephone number, organization, postal mail, purpose for which you will use the database, time and date at which you sent the email with the signed license agreement.
 

3) Once the email copy of the license agreement has been received at ATVS, you will receive an email with a username, a password, and a time slot to download the database.
 

4) [Download the benchmark](http://atvs.ii.uam.es/atvs/intranet/free_DB/beCAPTCHA), for which you will need to provide the authentication information given in step 4. After you finish the download, please notify by email to **atvs@uam.es** that you have successfully completed the transaction.
 

5) For more information, please contact: **atvs@uam.es**


## DESCRIPTION OF TypeNet benchmark
This benchmark contains the TypeNet embedding vectors from 130K subjects generated during natural typing in both touchscreen virtual (30K subjects) and physical keyboards (100K subjects) scenarios. Aditionally, we provide a experimental protocol to reproduce the results obtained in Acien *et al.* [1] paper.

**Keystroke Datasets**  
The embedding vectors are obtained when passing through TypeNet network the keystroke sequences acquired from the two Aalto University Datasets: 1) Dhakal *et al.* [2] dataset, which comprises more than 5GB of keystroke data collected in desktop keyboards from 168K participants; and 2) Palin  *et al.* [3] dataset, which comprises almost 4GB of keystroke data collected in mobile devices from 260K participants. The data were collected following the same procedure for both datasets. The acquisition task required subjects to memorize English sentences and then type them as quickly and accurate as they could. The English sentences were selected randomly from a set of 1525 examples taken from the Enron mobile email and Gigaword Newswire corpus. The example sentences contained a minimum of 3 words and a maximum of 70 characters. Note that the sentences typed by the participants could contain more than 70 characters because each participant could forget or add new characters when typing. All participants in the Dhakal database completed 15 sessions (i.e. one sentence for each session) on either a desktop or a laptop physical keyboard. However, in the Palin dataset the participants who finished at least 15 sessions are only 23% (60K participants) over 260 participants that started the typing test.

**TypeNet Architecture**  
The TypeNet architecture is depicted in Fig. 1. It is composed of two Long Short-Term Memory (LSTM) layers of 128 units (*tanh* activation function). Between the LSTM layers, we perform batch normalization and dropout at a rate of 0.5 to avoid overfitting. Additionally, each LSTM layer has a recurrent dropout rate of 0.2. 

In order to train TypeNet with sequences of different lengths *N* within a single batch, we truncate the end of the input sequence when *N>M* and zero pad at the end when *N<M*, in both cases to the fixed size *M*. The embedding vector provided are obtained for keystroke sequence of size *M=50* keys.
Finally, the output of the model **f(x)** is an array of size *1X128* that we will employ later as an embedding feature vectors to authenticate subjects.

![](https://github.com/BiDAlab/BeCAPTCHA-Mouse/blob/master/Fig6.png)
**Figure 2. The proposed architecture to train a GAN Generator of synthetic mouse trajectories.The Generator learns the human features of the mouse trajectories and generate human-like ones from Gaussian Noise.**

The aim of the Generator is to fool the Discriminator by generating synthetic mouse trajectories very similar to the real ones, while the Discriminator has to predict whether the sample comes from the real set or is a fake created by the Generator. Once the Generator is trained this way, then we can use it to synthesize mouse trajectories very similar to the human ones.
Fig. 1 shows two examples (trajectories B and C) of synthetic mouse trajectories generated with the GAN network and the comparison with a real one.  
The human mouse trajectories employed to train the GAN network were extracted from Chao *et al.* [2] database, which is comprised of more than 200K mouse trajectories acquired from 58 users who completed 300 repetitions of the task. In each repetition, the users had to click 8 buttons that appeared in the screen sequentially. This task was repeated twice in each session.


#### BENCHMARK STRUCTURE
BeCAPTCHA-Mouse benchmark are composed by two main folders: *'DB_GAN'* which contains the synthetic GAN trayectories and *'DB_fnc'* that contains the function-based ones. Each main folder has other two folders: *'raw'* folder which contains the raw data of the synthetic mouse trayectories in .txt files, and *'neuromotor'* folder that contains the Sigma-Lognormal descomposition (more details in [3]) of the raw files in .ana format. Both kind of files have the same name to match them easily.

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
