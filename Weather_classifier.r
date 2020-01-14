#PROGETTO MACHINE LEARNING 2019/2020

#-------- Descrizione del dominio di riferimento e obiettivi dell'elaborato ----------

# TO DO

#-------- Scelte di design per la creazione del data set, eventuali ipotesi o assunzioni ----------

setwd("C:/Users/Davide Finati/Desktop/Universita'/Magistrale/I anno/I semestre/Machine Learning/Progetto/2019_machinelearning")
setwd("/Users/fabiobeltramelli/Desktop/2019_machinelearning")

#INSTALLO E CARICO TUTTE LE LIBRERIE CHE VERRANNO UTILIZZATE
install.packages(c("caret", "mlbench", "rpart", "rpart.plot", "randomForest", "rattle", "RColorBrewer", "corrplot", "class", "FactoMineR", "factoextra", "e1071", "neuralnet")) 
library(caret)
library(mlbench)
library(rpart)
library(rpart.plot)
library(randomForest)
library(RColorBrewer)
library(corrplot)
library(class)
library(FactoMineR)
library(factoextra)
library(e1071)
library(neuralnet)

#CARICO IL DATASET COMPLETO
dataset <- read.csv("Dataset.csv", stringsAsFactors = T, sep = ',', header = TRUE)

#COME RICHESTO DAL PROPRIETARIO DEL DATASET E' STATO RIMOSSO L'ATTRIBUTO RISK_MM CHE E' IL TARGET DI UN IPOTETICO TASK DI REGRESSIONE
#SONO STATI RIMOSSI ANCHE GLI ATTRIBUTI RELATIVI ALLA DATA E ALLA LOCALITA' IN QUANTO CONSIDERATI SUPERFLUI
dataset <- dataset[-c(1:2,23)]

colnames(dataset)

#GLI ATTRIBUTI RAIN TODAY E RAIN TOMORROW SONO BOOLEANI, QUINDI TRADUCO LE STRINGHE "NO" E "YES" IN 0 E 1 RISPETTIVAMENTE
dataset$RainToday = ifelse(dataset$RainToday=="No", 0, 1)
dataset$RainTomorrow = ifelse(dataset$RainTomorrow=="No", 0, 1)

#TRADUCO COLONNA TARGET E RAIN TODAY IN FATTORI 
#dataset$RainToday = as.numeric(factor(dataset$RainToday))
#dataset$WindDir9am = as.numeric(factor(dataset$WindDir9am))
#dataset$WindDir3pm = as.numeric(factor(dataset$WindDir3pm))
#dataset$WindGustDir = as.numeric(factor(dataset$WindGustDir))
dataset$RainToday = factor(dataset$RainToday)
dataset$RainTomorrow = factor(dataset$RainTomorrow)

#INIZIO A VEDERE LA STRUTTURA GENERALE DEL DATASET
str(dataset)
head(dataset)

#MI ACCORGO CHE GIA' DALLE PRIME RIGHE DEL DATASET CHE ALCUNI ATTRIBUTI CONTENGONO MOLTI VALORI NA E QUINDI FACCIO UNA
#ANALISI PIU' APPROFONDITA:
for (i in c(1:ncol(dataset))){
  print(paste(colnames(dataset[i]), "| Number of NA: ", sum(is.na(dataset[i])), "| % of NA: ", sum(is.na(dataset[i]))/nrow(dataset)*100))
}
#L'ANALISI MI CONFERMA CHE SPECIALMENTE PER 4 COLONNE HO UN NUMERO DI NA MOLTO ELEVATO (DAL 37% AL 47%) QUINDI
#PER EVITARE DI INTRODURRE DATI ARTIFICIALI (MEDIA O ALTRO) SU COSI' TANTI VALORI LA STRATEGIA ATTUATA SARA' QUELLA DI RIMUOVERE
#QUESTI 4 ATTRIBUTI (EVAPORATION, SUNSHINE, CLOUD9AM, CLOUD3PM)

dataset = dataset[-c(4:5, 16:17)]
colnames(dataset)

#ORA AVENDO OSSERVATO CHE LE ALTRE NON HANNO UN NUMERO DI NA COSI' SIGNIFICATIVO LA STRATEGIA PIU' VELOCE E 
# CHE PERMETTE DI NON INTRODURRE ASSUNZIONI ESTERNE E' OMETTERE TUTTE LE OSSERVAZIONI CHE CONTENGONO NA
dataset = na.omit(dataset)
#IL DATASET PASSA PERO' DA 142193 OSSERVAZIONI A 112925 (ERANO PRESENTI QUINDI CIRCA 30000 OSSERVAZIONI CHE CONTENEVANO NA)

#ORA LA FUNZIONE STR MI INDICA ANCHE QUALI OSSERVAZIONI SONO STATE RIMOSSE ATTRAVERSO LA NA.OMIT (PER ESEMPIO
#L'OSSERVAZIONE NUMERO 15)
str(dataset)
head(dataset, n = 20)
for (i in c(1:ncol(dataset))){
  print(paste(colnames(dataset[i]), "| Number of NA: ", sum(is.na(dataset[i])), "| % of NA: ", sum(is.na(dataset[i]))/nrow(dataset)*100))
}

#PER CONFERMA CONTROLLO CHE L'OSSERVAZIONE 15 NON CI SIA PIU' E NON SIANO PIU' PRESENTI NA -> OK!

#-------- analisi esplorativa del training set (analisi delle covariate e/o PCA) ---------

#MATRICE DI CORRELAZIONE 
#CONSIDERO SOLO GLI ATTRIBUTI NUMERICI PER LA CORRELAZIONE

M <- cor(dataset[-c(4, 6:7, 16:17)])
M

#brewer.pal(n=8, name="RdBu")
#method = color, number o circle
col <- colorRampPalette(c("#BB4444", "#EE9988", "#FFFFFF", "#77AADD", "#4477AA"))
corrplot(M, method="color", col=col(200),
         type="upper", order="hclust", 
         addCoef.col = "black", # AGGIUNGE COEFFICENTE DI CORRELAZIONE
         tl.col="black", tl.srt=45, # COLORE E ROTAZIONE DELLA LABEL
         # NASCONDE LA CORRELAZIONE NELLA DIAGONALE PRINCIPALE (TUTTA = A 1)
         diag=F 
)

#FEATURE SELECTION:
#CALCOLO ATTRAVERSO LA FUNZIONE FINDCORRELATION QUALI SONO GLI ATTRIBUTI DA RIMUOVERE IN BASE ALLA LORO CORRELAZIONE
#MOTIVO: SE DUE ATTRIBUTI HANNO CORRELAZIONE MOLTO ALTA POSSO NON CONSIDERARNE UNO DEI DUE
highlyCorrelated <- findCorrelation(M, cutoff = 0.67, names = TRUE)
#VISUALIZZO I NOMI DEGLI ATTRIBUTI DA RIMUOVERE
highlyCorrelated
#RIMUOVO GLI ATTRIBUTI
dataset <- dataset[-c(2, 5, 10, 13:15)]

#VISUALIZZO LE COLONNE DEL DATASET RIDOTTO
colnames(dataset)

#PCA
#CALCOLO PCA
res.pca <- PCA(dataset[-c(3:5, 10:11)], scale.unit = TRUE, graph = FALSE)

#OTTENGO E MOSTRO GLI AUTOVALORI
eig.val <- get_eigenvalue(res.pca)
eig.val
#VALORI >1 MI INDICANO CHE LA DIMENSIONE CONSIDERATA HA IMPORTANZA E DEVE ESSERE CONSIDERATA (NEL NOSTRO CASO QUINDI
#CONSIDERIAMO LE PRIME 3 DIMENSIONI)

#ALTRIMENTI POSSIAMO ATTRAVERSO IL GRAFICO SEGUENTE CONSIDERARE DI ARRIVARE AD UNA CERTA "SOGLIA" DI QUANTO IL DATASET
#VIENE DESCRITTO
fviz_eig(res.pca, addlabels = TRUE, ylim = c(0, 50))
#PER ESEMPIO SE ANCORA UNA VOLTA CONSIDERASSI SOLO LE PRIME 3 DIMENSIONI QUESTE MI DESCRIVEREBBERO IL 70% DEL DATASET

#OTTENGO INFORMAZIONE DAL PCA SULLE VARIABILI
var <- get_pca_var(res.pca)
#IN PARTICOLARE MI POTREBBERO INTERESSARE LA CORRELAZIONE TRA LE VARIABILI E LE DIMENSIONI O IL LORO CONTRIBUTO
var$cor
var$contrib
#INOLTRE POSSO VEDERE QUANTO LE VARIABILI SONO CORRELATE TRA DI LORO ATTRAVERSO QUESTO GRAFICO
fviz_pca_var(res.pca, col.var = "black")

#POSSO EFFETTUARE DELLE ANALISI ANCHE DAL PCA SUGLI INDIVIDUI
ind <- get_pca_ind(res.pca)
ind

#DIVIDO IL DATASET IN TRAIN E TEST (70% E 30%)
set.seed(123)
ind = sample(2, nrow(dataset), replace = T,  prob = c(0.7, 0.3))
trainset = dataset[ind == 1,]
testset = dataset[ind == 2,]
#HO CREATO COSI' TRAINSET E TESTSET CHE VERRANNO UTILIZZATI PER ANALISI E ALLENARE E TESTARE I MODELLI CHE VERRANNO IMPLEMENTATI


#ANALISI INTUITIVA SULLA VARIABILE TARGET...
table(trainset$RainTomorrow)
prop.table(table(trainset$RainTomorrow))

plot(trainset$RainTomorrow, names = c("No", "Yes"), col=c("red", "green"))
#PER PRIMA COSA VEDO CHE IL TARGET SCELTO NEL TRAINSET NON E' BILANCIATO INFATTI IL 78% CIRCA DELLE 
#OSSERVAZIONI HANNO UN ETICHETTA NEGATIVA (CIOE' NON E' PREVISTA PIOGGIA PER IL GIORNO SUCCESSIVO)

#INTUITIVAMENTE POTREI PENSARE CHE SE QUEL GIORNO PIOVE ALLORA E' PROBABILE CHE PIOVA ANCHE IL SUCCESSIVO E QUINDI
#CONTROLLO CHE RELAZIONE HANNO I DUE ATTRIBUTI
table(trainset$RainTomorrow, trainset$RainToday)
prop.table(table(trainset$RainToday, trainset$RainTomorrow),1)
barplot(table(trainset$RainTomorrow, trainset$RainToday), col=c("red", "green"), names = c("No", "Yes"), legend=c("No","Yes"))
#IN REALTA' SCOPRO CHE E' MOLTO PROBABILE CHE SE UN GIORNO NON PIOVE (riga 0) ALLORA NON PIOVERA' ANCHE IN QUELLO SUCCESSIVO 
#(ANCHE SE QUESTO E' DATO DAL FATTO CHE IL TAGET E' SBILANCIATO) MA NON E' VERA LA MIA ASSUNZIONE IN QUANTO NOTO CHE
#SE IN UN GIORNO PIOVESSE (riga 1) E' PIU' PROBABILE CHE IL GIORNO DOPO NON CI SIA PIOGGIA

#-------- modelli scelti --------

#MODELLO 0: DUMMY MODEL

#L'ASSUNZIONE PIU' SEMPLICE DA FARE DOPO AVER ANALIZZATO IL TRAINSET E' CONSIDERARE CHE IL TARGET E' SBILANCIATO 
#SUL "NO" QUINDI EFFETTUO UNA PRIMA PREVISIONE DI TUTTI "NO" (0) CHE UTILIZZERO' POI COME BENCHMARK
testset$Prediction = 0
testset$Prediction = factor(testset$Prediction)

#UNA  VOLTA AGGIUNTA LA COLONNA DELLE PREVISIONI AL TRAINSET MISURO LE PERFORMANCE (CONF MATRIX E ACCURACY)
confMatrix <- table(testset$Prediction, testset$RainTomorrow)
confMatrix
print(paste("Accuracy: ", sum(diag(confMatrix))/sum(confMatrix)))

#OPPURE
confusionMatrix(testset$Prediction, testset$RainTomorrow)

#UN ACCURACY DI 77.6%, DECISAMENTE NON MALE VISTO CHE NON ABBIAMO INTRDOTTO NESSUN MODELLO E NON ABBIAMO UTILIZZATO
#RISORSE. 
#N.B. QUESTO RISULATATO COSI' BUONO E' IL FRUTTO DELLA VARIABILE TARGET SBILANCIATA!!!!!
#IL NOSTRO OBIETTIVO SARA' QUINDI QUELLO DI CREARE DEI MODELLI PREDITTIVI CHE MIGLIORINO IL 77% DI QUESTO PRIMO APPROCCIO BASE

#COME PROVA DELLE ANALISI FATTE IN PRECEDENZA POSSIAMO VEDERE QUALI SONO LE PERFORMANCE SE AL POSTO DI TUTTI 0
#IMPONGO CHE SE IN UN DATO GIORNO PIOVE ALLORA PREVEDO CHE CI SARA' PIOGGIA ANCHE IN QUELLO SUCCESSIVO
#REMINDER: ERA STATO DETTO CHE QUESTA ASSUNZIONE NON E' VERIFICATA...
testset$Prediction = 0
testset$Prediction[testset$RainToday == 1] = 1
testset$Prediction = factor(testset$Prediction)

confusionMatrix(testset$Prediction, testset$RainTomorrow)
#TALE ASSUNZIONE RISULTA ANCORA UNA VOLTA NON FONDATA IN QUANTO LE PERFORMANCE DEL MODELLO SONO DIMINUITE RISPETTO
#AL PRIMO APPROCCIO (CIOE' QUELLO CON TUTTI 0) (DA 77.6% A 76.4%)


#MODELLO 1: DECISION TREE / RANDOM FOREST

#PER PRIMA COSA COSTRUISCO UN ALBERO DI DECISIONE LIBERO DI CRESCERE LIBERAMENTE
start_time <- Sys.time()
bigDecisionTree = rpart(RainTomorrow ~ ., 
                        data = trainset, method = "class", control = rpart.control(cp = 0))
end_time <- Sys.time()

print(end_time - start_time)

#VISUALIZZO GRAFICAMENTE IL PLOT DELL'ALBERO E DELLA SUA COMPLESSITA' DEI PARAMETRI
plot(bigDecisionTree)
plotcp(bigDecisionTree)
#MI ACCORGO SUBITO CHE OLTRE UN CERTO PUNTO OLTRE CHE DIVENTARE SEMPRE PIU' ONEROSO IN TERMINI COMPUTAZIONALI 
#L'ALBERO NON MIGLIORA PIU' DI UN CERTO LIMIRE

#CALCOLO LA PREDICTION UTILIZZANDO QUESTO MODELLO
start_time <- Sys.time()
testset$Prediction <- predict(bigDecisionTree, testset, type = "class")
end_time <- Sys.time()

print(end_time - start_time)

#CALCOLO LE PERFORMANCE DEL MODELLO
confMatrix = table(testset$Prediction, testset$RainTomorrow)
confMatrix
print(paste("Accuracy Big Decision Tree: ", sum(diag(confMatrix))/sum(confMatrix)))
#VEDO CHE COMUNQUE L'ACCURACY E' MIGLIORE RISPETTO AL PRIMO MODELLO DUMMY (DA 77.6% A 81% CIRCA), DEVO CERCARE PERO' DI OTTIMIZZARE IL MIO ALBERO

#VOGLIO CAPIRE COME DI COMPORTA SE LIMITO AD UN VALORE RAGIONEVOLE LA PROFONDITA' MASSIMA DI CRESCITA DELL'ALBERO DI DECISIONE
start_time <- Sys.time()
decisionTree = rpart(RainTomorrow ~ ., 
                     data = trainset, method = "class", control = rpart.control(cp = 0, maxdepth = 6))
end_time <- Sys.time()

print(end_time - start_time)

#HO CREATO IL MODELLO ALLENATO SUL TRAINSET

#VISUALIZZO GRAFICAMENTE IL PLOT DELL'ALBERO E DELLA SUA COMPLESSITA' DEI PARAMETRI
plot(decisionTree)
plotcp(decisionTree)
#NOTO GIA' "A OCCHIO" CHE LA COMPLESSITA' DELL'ALBERO E' DIMINUITA

#CALCOLO LA PREDICTION UTILIZZANDO QUESTO MODELLO
start_time <- Sys.time()
testset$Prediction <- predict(decisionTree, testset, type = "class")
end_time <- Sys.time()

print(end_time - start_time)

#CALCOLO LE PERFORMANCE DEL MODELLO
confMatrix = table(testset$Prediction, testset$RainTomorrow)
confMatrix
print(paste("Accuracy Decision Tree: ", sum(diag(confMatrix))/sum(confMatrix)))
#DALLE PERFORMANCE VEDO ANCHE CHE L'ACCURACY E' MIGLIORATA PASSANDO DA 81% CIRCA A 84%

#MI ACCORGO PERO' CHE POTREI RENDERE IL MODELLO ANCORA PIU' "SNELLO" E COMPUTAZIONALMENTE PIU' EFFICIENTE
#UTILIZZO LA FUNZIONE "prune" PER POTARE L'ALBERO, QUESTA EVITA IL FENOMENO DELL'OVERFITTING E RIDUCE LA COMPLESSITA'
#PER SCEGLIERE UN CP OPPORTUNO GUARDO IL GRAFICO PLOTCP DEL DECISION TREE CHE VOGLIO POTARE
start_time <- Sys.time()
prunedDecisionTree = prune(decisionTree, cp = 0.017)
end_time <- Sys.time()

print(end_time - start_time)

#CREO UN ALBERO POTATO

#ESSENDO MOLTO PIU' RIDOTTO POSSO VISUALIZZARLO IN MODO MIGLIORE
fancyRpartPlot(prunedDecisionTree)
#VEDO CHE IL MODELLO CREATO UTILIZZA AL MASSIMO 3 ATTRIBUTI PER ASSOCIARE UN ETICHETTA ALLE OSSERVAZIONI,
#VOGLIO PERO' CONTROLLARE L'ACCURACY CHE MI GARANTISCE QUESTO MODELLO "SEMPLIFICATO"

#CALCOLO LA PREDICTION UTILIZZANDO QUESTO MODELLO SEMPLIFICATO
start_time <- Sys.time()
testset$Prediction <- predict(prunedDecisionTree, testset, type = "class")
end_time <- Sys.time()

print(end_time - start_time)

#CALCOLO LE PERFORMANCE DEL MODELLO
confMatrix = table(testset$Prediction, testset$RainTomorrow)
confMatrix
print(paste("Accuracy Pruned Decision Tree: ", sum(diag(confMatrix))/sum(confMatrix)))
#HO UNA ACCURACY DEL 83% CIRCA E' LEGGERMENTE MINORE RISPETTO AL MODELLO PRECEDENTE (84%) MA MOLTO MIGLIORE IN TERMINI 
#DI COMPLESSITA' E QUINDI E' IL MODELLO DA PREFERIRE!!

#ALLENO ORA UNA RANDOM FOREST
start_time <- Sys.time()
randomForest = randomForest(RainTomorrow ~ ., 
                            data = trainset, method = "class", ntree = 500)
end_time <- Sys.time()

print(end_time - start_time)

#TALE MODELLO E' DECISAMENTE PIù LENTO DI DECISION TREE PERCHE' ALLENA DIVERSI ALBERI DECISIONALI PER POI TENERE IL MIGLIORE
#TEMPO: 1/2 MINUTI

#CREO LA PREVISIONE UTILIZZANDO IL MODELLO ALLENATO
start_time <- Sys.time()
testset$Prediction <- predict(randomForest, testset, type = "class")
end_time <- Sys.time()

print(end_time - start_time)

#CALCOLO LE PERFORMANCE DEL MODELLO
confMatrix = table(testset$Prediction, testset$RainTomorrow)
confMatrix
print(paste("Accuracy Random Forest: ", sum(diag(confMatrix))/sum(confMatrix)))
#SI PUO' NOTARE PERO' CHE IL MODELLO E' PIU' PRECISO ARRIVANDO A SFIORARE L'85% DI ACCURACY

#MODELLO 2: NAIVE BAYES
start_time <- Sys.time()
naiveBayes = naiveBayes(trainset, trainset$RainTomorrow)
end_time <- Sys.time()

print(end_time - start_time)

#CREO LA PREVISIONE UTILIZZANDO IL MODELLO ALLENATO
start_time <- Sys.time()
testset$Prediction <- predict(naiveBayes, testset, type = "class")
end_time <- Sys.time()

print(end_time - start_time)

#CALCOLO LE PERFORMANCE DEL MODELLO
confMatrix = table(testset$Prediction, testset$RainTomorrow)
confMatrix
print(paste("Accuracy Naive Bayes: ", sum(diag(confMatrix))/sum(confMatrix)))
#NAIVE BAYES MI DA UN ACCURACY ALTISSIMA, SUPERIORE AL 99%

#MODELLO 3: NEURAL NETWORK
nn = neuralnet(RainTomorrow ~ MinTemp + Rainfall + WindSpeed9am + WindSpeed3pm + Humidity3pm + Pressure9am,
               trainset, hidden = length(dataset))
plot(nn)

#CREO LA PREVISIONE UTILIZZANDO IL MODELLO ALLENATO
#testset$Prediction <- predict(nn, testset)

testset$Prediction <- compute(nn, testset)
length(testset$Prediction)

#CALCOLO LE PERFORMANCE DEL MODELLO
confMatrix = table(testset$Prediction, testset$RainTomorrow)
confMatrix
print(paste("Accuracy Neural Network: ", sum(diag(confMatrix))/sum(confMatrix)))

#MODELLO 4: SVM
start_time <- Sys.time()
svm = svm(RainTomorrow ~ ., data = trainset, kernel = 'linear', scale = TRUE)
end_time <- Sys.time()

print(end_time - start_time)

#CREO LA PREVISIONE UTILIZZANDO IL MODELLO ALLENATO
start_time <- Sys.time()
testset$Prediction <- predict(svm, testset, type = "class")
end_time <- Sys.time()

print(end_time - start_time)

#CALCOLO LE PERFORMANCE DEL MODELLO
confMatrix = table(testset$Prediction, testset$RainTomorrow)
confMatrix
print(paste("Accuracy SVM: ", sum(diag(confMatrix))/sum(confMatrix)))
#CON UN PRIMO APPROCCIO IL MODELLO SVM HA UNA ACCURACY DEL 84% CIRCA IMPIEGANDOCI PERO' MOLTO PIU' TEMPO DEI MODELLI
#PRECEDENTI (CIRCA 15 MINUTI PER ALLENARE IL MODELLO)

#PROVIAMO KERNEL DIVERSI
start_time <- Sys.time()
svm = svm(RainTomorrow ~ ., data = trainset, kernel = 'radial', scale = TRUE)
end_time <- Sys.time()

print(end_time - start_time)

#CREO LA PREVISIONE UTILIZZANDO IL MODELLO ALLENATO
start_time <- Sys.time()
testset$Prediction <- predict(svm, testset, type = "class")
end_time <- Sys.time()

print(end_time - start_time)

#CALCOLO LE PERFORMANCE DEL MODELLO
confMatrix = table(testset$Prediction, testset$RainTomorrow)
confMatrix
print(paste("Accuracy SVM: ", sum(diag(confMatrix))/sum(confMatrix)))

#KERNEL: 
#1)LINEAR: 14 MINUTI - 84% ACCURACY
#2)POLYNOMIAL: 8 MINUTI - 82.5% ACCURACY
#3)RADIAL: 15 MINUTI - 84% ACCURACY
#4)SIGMOID: 6 MINUTI - 79% ACCURACY