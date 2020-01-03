#PROGETTO MACHINE LEARNING 2019/2020

#-------- Descrizione del dominio di riferimento e obiettivi dell'elaborato ----------

# TO DO

#-------- Scelte di design per la creazione del data set, eventuali ipotesi o assunzioni ----------

#INSTALLO E CARICO TUTTE LE LIBRERIE CHE VERRANNO UTILIZZATE
install.packages(c("caret", "rpart", "randomForest", "rattle", "RColorBrewer", "corrplot")) 
library(caret)
library(rpart)
library(rpart.plot)
library(randomForest)
library(rattle)
library(RColorBrewer)
library(corrplot)


#CARICO IL DATASET COMPLETO
dataset <- read.csv("Dataset.csv", stringsAsFactors= F)

#COME RICHESTO DAL PROPRIETARIO DEL DATASET NON CONSIDERO L'ATTRIBUTO MMRISK CHE E' IL TARGET DI UN IPOTETICO TASK DI REGRESSIONE
dataset <- dataset[-23]

#GLI ATTRIBUTI RAIN TODAY E RAIN TOMORROW SONO BOOLEANI, QUINDI TRADUCO LE STRINGHE "NO" E "YES" IN 0 E 1 RISPETTIVAMENTE
dataset$RainToday = ifelse(dataset$RainToday=="No", 0, 1)
dataset$RainTomorrow = ifelse(dataset$RainTomorrow=="No", 0, 1)

#TRADUCO COLONNA TARGET E RAIN TODAY IN FATTORI E DATA COME DATE 
dataset$RainToday = factor(dataset$RainToday)
dataset$RainTomorrow = factor(dataset$RainTomorrow)
dataset$Date = as.Date(dataset$Date)

#INIZIO A VEDERE LA STRUTTURA GENERALE DEL DATASET
str(dataset)
head(dataset)

#MI ACCORGO CHE GIA' DALLE PRIME RIGHE DEL DATASET CHE ALCUNI ATTRIBUTI CONTENGONO MOLTI VALORI NA E QUINDI FACCIO UNA
#ANALISI PIU' APPROFONDITA:
for (i in c(1:ncol(dataset))) {
  print(paste(colnames(dataset[i]), "| Number of NA: ", sum(is.na(dataset[i])), "| % of NA: ", sum(is.na(dataset[i]))/nrow(dataset)*100))
}
#L'ANALISI MI CONFERMA CHE SPECIALMENTE PER 4 COLONNE HO UN NUMERO DI NA MOLTO ELEVATO (DAL 37% AL 47%) QUINDI
#PER EVITARE DI INTRODURRE DATI ARTIFICIALI (MEDIA O ALTRO) SU COSI' TANTI VALORI LA STRATEGIA ATTUATA SARA' QUELLA DI RIMUOVERE
#QUESTI 4 ATTRIBUTI (EVAPORATION, SUNSHINE, CLOUD9AM, CLOUD3PM)

dataset = dataset[c(1:5, 8:17, 20:23)]
#only numeric attributes
#dataset = dataset[c(3:5, 9, 12:17, 20:23)]

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

#DIVIDO IL DATASET IN TRAIN E TEST (70% E 30%)
set.seed(123)
ind = sample(2, nrow(dataset), replace = T,  prob = c(0.7, 0.3))
trainset = dataset[ind == 1,]
testset = dataset[ind == 2,]
#HO CREATO COSI' TRAINSET E TESTSET CHE VERRANNO UTILIZZATI PER ANALISI E ALLENARE E TESTARE I MODELLI CHE VERRANNO IMPLEMENTATI

#-------- analisi esplorativa del training set (analisi delle covariate e/o PCA) ---------

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

#!? ANALISI COVARIATE SI INTENDE COME QUELLA APPENA FATTA CON TUTTE LE VARIABILI O MATRICE DI CORRELLAZIONE!?

#MATRICE DI CORRELAZIONE 
#CONSIDERO SOLO GLI ATTRIBUTI NUMERICI PER LA CORRELAZIONE
M<-cor(trainset[c(3:5, 7, 10:17)])

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


#PCA (TO DO)


#-------- modelli scelti --------

#MODELLO 0: DUMMY MODEL

#L'ASSUNZIONE PIU' SEMPLICE DA FARE DOPO AVER ANALIZZATO IL TRAINSET E' CONSIDERARE CHE IL TARGET E' SBILANCIATO 
#SUL "NO" QUINDI EFFETTUO UNA PRIMA PREVISIONE DI TUTTI "NO" (0) CHE UTILIZZERO' POI COME BENCHMARK
testset$Prediction = rep(0, 33830)
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
bigDecisionTree = rpart(RainTomorrow ~ MinTemp + MaxTemp + Rainfall + WindGustSpeed + WindSpeed9am + WindSpeed3pm + Humidity9am + Humidity3pm + Pressure9am + Pressure3pm + Temp9am + Temp3pm + RainToday, 
                     data = trainset, method = "class", control = rpart.control(cp = 0))

#VISUALIZZO GRAFICAMENTE IL PLOT DELL'ALBERO E DELLA SUA COMPLESSITA' DEI PARAMETRI
plot(bigDecisionTree)
plotcp(bigDecisionTree)
#MI ACCORGO SUBITO CHE OLTRE UN CERTO PUNTO OLTRE CHE DIVENTARE SEMPRE PIU' ONEROSO IN TERMINI COMPUTAZIONALI 
#L'ALBERO NON MIGLIORA PIU' DI UN CERTO LIMIRE

#CALCOLO LA PREDICTION UTILIZZANDO QUESTO MODELLO
testset$Prediction <- predict(bigDecisionTree, testset, type = "class")

#CALCOLO LE PERFORMANCE DEL MODELLO
confMatrix = table(testset$Prediction, testset$RainTomorrow)
confMatrix
print(paste("Accuracy big decision tree: ", sum(diag(confMatrix))/sum(confMatrix)))
#VEDO CHE COMUNQUE L'ACCURACY E' MIGLIORE RISPETTO AL PRIMO MODELLO DUMMY (DA 77.6% A 82% CIRCA), DEVO CERCARE PERO' DI OTTIMIZZARE IL MIO ALBERO

#VOGLIO CAPIRE COME DI COMPORTA SE LIMITO LA PROFONDITA' MASSIMA DI CRESCITA DELL'ALBERO DI DECISIONE
decisionTree = rpart(RainTomorrow ~ MinTemp + MaxTemp + Rainfall + WindGustSpeed + WindSpeed9am + WindSpeed3pm + Humidity9am + Humidity3pm + Pressure9am + Pressure3pm + Temp9am + Temp3pm + RainToday, 
                     data = trainset, method = "class", control = rpart.control(cp = 0, maxdepth = 6))
#HO CREATO IL MODELLO ALLENATO SUL TRAINSET

#VISUALIZZO GRAFICAMENTE IL PLOT DELL'ALBERO E DELLA SUA COMPLESSITA' DEI PARAMETRI
plot(decisionTree)
plotcp(decisionTree)
#NOTO GIA' "A OCCHIO" CHE LA COMPLESSITA' DELL'ALBERO E' DIMINUITA

#CALCOLO LA PREDICTION UTILIZZANDO QUESTO MODELLO
testset$Prediction <- predict(decisionTree, testset, type = "class")

#CALCOLO LE PERFORMANCE DEL MODELLO
confMatrix = table(testset$Prediction, testset$RainTomorrow)
confMatrix
print(paste("Accuracy decision tree: ", sum(diag(confMatrix))/sum(confMatrix)))
#DALLE PERFORMANCE VEDO ANCHE CHE L'ACCURACY E' MIGLIORATA PASSANDO DA 82% CIRCA A 84%

#MI ACCORGO PERO' CHE POTREI RENDERE IL MODELLO ANCORA PIU' "SNELLO" E COMPUTAZIONALMENTE PIU' EFFICIENTE
#UTILIZZO LA FUNZIONE "prune" PER POTARE L'ALBERO, QUESTA EVITA IL FENOMENO DELL'OVERFITTING E RIDUCE LA COMPLESSITA'
#PER SCEGLIERE UN CP OPPORTUNO GUARDO IL GRAFICO PLOTCP DEL DECISION TREE CHE VOGLIO POTARE
prunedDecisionTree = prune(decisionTree, cp = 0.017)
#CREO UN ALBERO POTATO

#ESSENDO MOLTO PIU' RIDOTTO POSSO VISUALIZZARLO IN MODO MIGLIORE
fancyRpartPlot(prunedDecisionTree)
#VEDO CHE IL MODELLO CREATO UTILIZZA AL MASSIMO 3 ATTRIBUTI PER ASSOCIARE UN ETICHETTA ALLE OSSERVAZIONI,
#VOGLIO PERO' CONTROLLARE L'ACCURACY CHE MI GARANTISCE QUESTO MODELLO "SEMPLIFICATO"

#CALCOLO LA PREDICTION UTILIZZANDO QUESTO MODELLO SEMPLIFICATO
testset$Prediction <- predict(prunedDecisionTree, testset, type = "class")

#CALCOLO LE PERFORMANCE DEL MODELLO
confMatrix = table(testset$Prediction, testset$RainTomorrow)
confMatrix
print(paste("Accuracy pruned decision tree: ", sum(diag(confMatrix))/sum(confMatrix)))
#HO UNA ACCURACY DI 83.6% E' LEGGERMENTE MINORE RISPETTO AL MODELLO PRECEDENTE (84%) MA MOLTO MIGLIORE IN TERMINI 
#DI COMPLESSITA' E QUINDI E' IL MODELLO DA PREFERIRE!!

#ALLENO ORA UNA RANDOM FOREST
randomForest = randomForest(RainTomorrow ~ MinTemp + MaxTemp + Rainfall + WindGustSpeed + WindSpeed9am + WindSpeed3pm + Humidity9am + Humidity3pm + Pressure9am + Pressure3pm + Temp9am + Temp3pm + RainToday, 
                            data = trainset, method = "class")

#CREO LA PREVISIONE UTILIZZANDO IL MODELLO ALLENATO
testset$Prediction <- predict(randomForest, testset, type = "class")

#CALCOLO LE PERFORMANCE DEL MODELLO
confMatrix = table(testset$Prediction, testset$RainTomorrow)
confMatrix
print(paste("Accuracy random forest: ", sum(diag(confMatrix))/sum(confMatrix)))
