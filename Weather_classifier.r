#PROGETTO MACHINE LEARNING 2019/2020

#INSTALLO E CARICO TUTTE LE LIBRERIE CHE VERRANNO UTILIZZATE
install.packages(c("caret", "mlbench", "rpart", "rpart.plot", "randomForest", "rattle", "RColorBrewer", "corrplot", "class", "FactoMineR", "factoextra", "e1071", "nnet", "doParallel", "pROC", "fitdistrplus")) 
library(caret)
library(mlbench)
library(rpart)
library(rpart.plot)
library(randomForest)
library(rattle)
library(RColorBrewer)
library(corrplot)
library(class)
library(FactoMineR)
library(factoextra)
library(e1071)
library(nnet)
library(doParallel)
library(pROC)
library(fitdistrplus)

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

#PROVO A VEDERE A QUALI DISTRIBUZIONI ASSOMIGLIANO I DATI PRESENTI NEL DATASET ED ESPLORO I VALORI DEGLI ATTRIBUTI

#CONSIDERO MINTEMP E AD OCCHIO SEMBRA AVVICINARSI ALLA DISTRIBUZIONE GAUSSIANA
q = quantile(dataset$MinTemp)
hist(dataset$MinTemp, main = "Distribuzione MinTemp", xlab = "MinTemp", freq = TRUE)
abline(v = q[1], col = "red", lwd = 2) # 0% (min)
abline(v = q[2], col = "blue", lwd = 2) # 1st quartile 25%
abline(v = q[3], col = "green", lwd = 2.5) # Median value distribution 50%
abline(v = q[4], col = "blue", lwd = 2) # 3rd quartile 75%
abline(v = q[5], col = "red", lwd = 2) # 100% (max)
boxplot(dataset$MinTemp, main = "Distribuzione MinTemp", xlab = "MinTemp")
hist(dataset$MinTemp, main = "Distribuzione MinTemp", xlab = "MinTemp", freq = FALSE)
lines(density(dataset$MinTemp), col = "red")
#CONTROLLO CHE L'ASSUNZIONE SIA VALIDA E OSSERVO IL VALORE MINIMO, MASSIMO, MEDIA E DEVIAZIONE STANDARD
descdist(dataset$MinTemp, discrete = FALSE)
fit.norm <- fitdist(dataset$MinTemp, "norm")
plot(fit.norm)

#CONSIDERO MAXTEMP E AD OCCHIO SEMBRA AVVICINARSI ALLA DISTRIBUZIONE GAUSSIANA
q = quantile(dataset$MaxTemp)
hist(dataset$MaxTemp, main = "Distribuzione MaxTemp", xlab = "MaxTemp")
abline(v = q[1], col = "red", lwd = 2) # 0% (min)
abline(v = q[2], col = "blue", lwd = 2) # 1st quartile 25%
abline(v = q[3], col = "green", lwd = 2.5) # Median value distribution 50%
abline(v = q[4], col = "blue", lwd = 2) # 3rd quartile 75%
abline(v = q[5], col = "red", lwd = 2) # 100% (max)
boxplot(dataset$MaxTemp, main = "Distribuzione MaxTemp", xlab = "MaxTemp")
hist(dataset$MaxTemp, main = "Distribuzione MaxTemp", xlab = "MaxTemp", freq = FALSE)
lines(density(dataset$MaxTemp), col = "red")
#CONTROLLO CHE L'ASSUNZIONE SIA VALIDA E OSSERVO IL VALORE MINIMO, MASSIMO, MEDIA E DEVIAZIONE STANDARD
descdist(dataset$MaxTemp, discrete = FALSE)
fit.norm <- fitdist(dataset$MaxTemp, "norm")
plot(fit.norm)

#CONSIDERO RAINFALL E NOTO CHE I VALORI SONO QUASI TUTTI A ZERO (64.5%)
hist(dataset$Rainfall, title = "Rainfall", sub = "")

#CONSIDERO WINDGUSTDIR
plot(dataset$WindGustDir, main = "Distribuzione WindGustDir", xlab = "WindGustDir")
table(dataset$WindGustDir)

#CONSIDERO WINDGUSTSPEED E AD OCCHIO SEMBRA AVVICINARSI ALLA DISTRIBUZIONE GAUSSIANA
q = quantile(dataset$WindGustSpeed)
hist(dataset$WindGustSpeed, main = "Distribuzione WindGustSpeed", xlab = "WindGustSpeed")
abline(v = q[1], col = "red", lwd = 2) # 0% (min)
abline(v = q[2], col = "blue", lwd = 2) # 1st quartile 25%
abline(v = q[3], col = "green", lwd = 2.5) # Median value distribution 50%
abline(v = q[4], col = "blue", lwd = 2) # 3rd quartile 75%
abline(v = q[5], col = "red", lwd = 2) # 100% (max)
boxplot(dataset$WindGustSpeed, main = "Distribuzione WindGustSpeed", xlab = "WindGustSpeed")
hist(dataset$WindGustSpeed, main = "Distribuzione WindGustSpeed", xlab = "WindGustSpeed", freq = FALSE)
lines(density(dataset$WindGustSpeed), col = "red")
#CONTROLLO CHE L'ASSUNZIONE SIA VALIDA E OSSERVO IL VALORE MINIMO, MASSIMO, MEDIA E DEVIAZIONE STANDARD
descdist(dataset$WindGustSpeed, discrete = FALSE)
#SEMBRA CHE UNA DISTRIBUZIONE APPROPRIATA SIA LA LOGISTICA
fit.lognormal <- fitdist(dataset$WindGustSpeed, "logis")
plot(fit.lognormal)

#CONSIDERO WINDDIR9AM
plot(dataset$WindDir9am, main = "Distribuzione WindDir9am", xlab = "WindDir9am")
table(dataset$WindDir9am)

#CONSIDERO WINDDIR3PM
plot(dataset$WindDir3pm, main = "Distribuzione WindDir3pm", xlab = "WindDir3pm")
table(dataset$WindDir3pm)

#CONSIDERO WINDSPEED9AM
q = quantile(dataset$WindSpeed9am)
hist(dataset$WindSpeed9am, main = "Distribuzione WindSpeed9am", xlab = "WindSpeed9am")
abline(v = q[1], col = "red", lwd = 2) # 0% (min)
abline(v = q[2], col = "blue", lwd = 2) # 1st quartile 25%
abline(v = q[3], col = "green", lwd = 2.5) # Median value distribution 50%
abline(v = q[4], col = "blue", lwd = 2) # 3rd quartile 75%
abline(v = q[5], col = "red", lwd = 2) # 100% (max)
boxplot(dataset$WindSpeed9am, main = "Distribuzione WindSpeed9am", xlab = "WindSpeed9am")
hist(dataset$WindSpeed9am, main = "Distribuzione WindSpeed9am", xlab = "WindSpeed9am", freq = FALSE)
lines(density(dataset$WindSpeed9am), col = "red")
#OSSERVO IL VALORE MINIMO, MASSIMO, MEDIA E DEVIAZIONE STANDARD
descdist(dataset$WindSpeed9am, discrete = FALSE)
#SEMBRA CHE UNA DISTRIBUZIONE APPROPRIATA SIA LA GAMMA
fit.gamma <- fitdist(dataset$WindSpeed9am, "gamma")
plot(fit.gamma)

#CONSIDERO WINDSPEED3PM
q = quantile(dataset$WindSpeed3pm)
hist(dataset$WindSpeed3pm, main = "Distribuzione WindSpeed3pm", xlab = "WindSpeed3pm")
abline(v = q[1], col = "red", lwd = 2) # 0% (min)
abline(v = q[2], col = "blue", lwd = 2) # 1st quartile 25%
abline(v = q[3], col = "green", lwd = 2.5) # Median value distribution 50%
abline(v = q[4], col = "blue", lwd = 2) # 3rd quartile 75%
abline(v = q[5], col = "red", lwd = 2) # 100% (max)
boxplot(dataset$WindSpeed3pm, main = "Distribuzione WindSpeed3pm", xlab = "WindSpeed3pm")
hist(dataset$WindSpeed3pm, main = "Distribuzione WindSpeed3pm", xlab = "WindSpeed3pm", freq = FALSE)
lines(density(dataset$WindSpeed3pm), col = "red")
#OSSERVO IL VALORE MINIMO, MASSIMO, MEDIA E DEVIAZIONE STANDARD
descdist(dataset$WindSpeed3pm, discrete = FALSE)
#SEMBRA CHE UNA DISTRIBUZIONE APPROPRIATA SIA LA GAMMA
fit.gamma <- fitdist(dataset$WindSpeed3pm, "gamma")
plot(fit.gamma)

#CONSIDERO HUMIDITY9AM
q = quantile(dataset$Humidity9am)
hist(dataset$Humidity9am, main = "Distribuzione Humidity9am", xlab = "Humidity9am")
abline(v = q[1], col = "red", lwd = 2) # 0% (min)
abline(v = q[2], col = "blue", lwd = 2) # 1st quartile 25%
abline(v = q[3], col = "green", lwd = 2.5) # Median value distribution 50%
abline(v = q[4], col = "blue", lwd = 2) # 3rd quartile 75%
abline(v = q[5], col = "red", lwd = 2) # 100% (max)
boxplot(dataset$Humidity9am, main = "Distribuzione Humidity9am", xlab = "Humidity9am")
hist(dataset$Humidity9am, main = "Distribuzione Humidity9am", xlab = "Humidity9am", freq = FALSE)
lines(density(dataset$Humidity9am), col = "red")
#OSSERVO IL VALORE MINIMO, MASSIMO, MEDIA E DEVIAZIONE STANDARD
descdist(dataset$Humidity9am, discrete = FALSE)
#SEMBRA CHE UNA DISTRIBUZIONE APPROPRIATA SIA LA NORMALE
fit.norm <- fitdist(dataset$Humidity9am, "norm")
plot(fit.norm)

#CONSIDERO HUMIDITY3PM, SEMBRA RAGIONEVOLE PENSARE CHE SI AVVICINI A UNA NORMALE
q = quantile(dataset$Humidity3pm)
hist(dataset$Humidity3pm, main = "Distribuzione Humidity3pm", xlab = "Humidity3pm")
abline(v = q[1], col = "red", lwd = 2) # 0% (min)
abline(v = q[2], col = "blue", lwd = 2) # 1st quartile 25%
abline(v = q[3], col = "green", lwd = 2.5) # Median value distribution 50%
abline(v = q[4], col = "blue", lwd = 2) # 3rd quartile 75%
abline(v = q[5], col = "red", lwd = 2) # 100% (max)
boxplot(dataset$Humidity3pm, main = "Distribuzione Humidity3pm", xlab = "Humidity3pm")
hist(dataset$Humidity3pm, main = "Distribuzione Humidity3pm", xlab = "Humidity3pm", freq = FALSE)
lines(density(dataset$Humidity3pm), col = "red")
#CONTROLLO CHE L'ASSUNZIONE SIA VALIDA E OSSERVO IL VALORE MINIMO, MASSIMO, MEDIA E DEVIAZIONE STANDARD
descdist(dataset$Humidity3pm, discrete = FALSE)
#SEMBRA CHE UNA DISTRIBUZIONE APPROPRIATA SIA LA NORMALE
fit.norm <- fitdist(dataset$Humidity3pm, "norm")
plot(fit.norm)

#CONSIDERO PRESSURE9AM, SEMBRA RAGIONEVOLE PENSARE CHE SI AVVICINI A UNA NORMALE
q = quantile(dataset$Pressure9am)
hist(dataset$Pressure9am, main = "Distribuzione Pressure9am", xlab = "Pressure9am")
abline(v = q[1], col = "red", lwd = 2) # 0% (min)
abline(v = q[2], col = "blue", lwd = 2) # 1st quartile 25%
abline(v = q[3], col = "green", lwd = 2.5) # Median value distribution 50%
abline(v = q[4], col = "blue", lwd = 2) # 3rd quartile 75%
abline(v = q[5], col = "red", lwd = 2) # 100% (max)
boxplot(dataset$Pressure9am, main = "Distribuzione Pressure9am", xlab = "Pressure9am")
hist(dataset$Pressure9am, main = "Distribuzione Pressure9am", xlab = "Pressure9am", freq = FALSE)
lines(density(dataset$Pressure9am), col = "red")
#CONTROLLO CHE L'ASSUNZIONE SIA VALIDA E OSSERVO IL VALORE MINIMO, MASSIMO, MEDIA E DEVIAZIONE STANDARD
descdist(dataset$Pressure9am, discrete = FALSE)
#SEMBRA CHE UNA DISTRIBUZIONE APPROPRIATA SIA LA NORMALE
fit.norm <- fitdist(dataset$Pressure9am, "norm")
plot(fit.norm)

#CONSIDERO PRESSURE3PM, SEMBRA RAGIONEVOLE PENSARE CHE SI AVVICINI A UNA NORMALE
q = quantile(dataset$Pressure3pm)
hist(dataset$Pressure3pm, main = "Distribuzione Pressure3pm", xlab = "Pressure3pm")
abline(v = q[1], col = "red", lwd = 2) # 0% (min)
abline(v = q[2], col = "blue", lwd = 2) # 1st quartile 25%
abline(v = q[3], col = "green", lwd = 2.5) # Median value distribution 50%
abline(v = q[4], col = "blue", lwd = 2) # 3rd quartile 75%
abline(v = q[5], col = "red", lwd = 2) # 100% (max)
boxplot(dataset$Pressure3pm, main = "Distribuzione Pressure3pm", xlab = "Pressure3pm")
hist(dataset$Pressure3pm, main = "Distribuzione Pressure3pm", xlab = "Pressure3pm", freq = FALSE)
lines(density(dataset$Pressure3pm), col = "red")
#CONTROLLO CHE L'ASSUNZIONE SIA VALIDA E OSSERVO IL VALORE MINIMO, MASSIMO, MEDIA E DEVIAZIONE STANDARD
descdist(dataset$Pressure3pm, discrete = FALSE)
#SEMBRA CHE UNA DISTRIBUZIONE APPROPRIATA SIA LA NORMALE
fit.norm <- fitdist(dataset$Pressure3pm, "norm")
plot(fit.norm)

#CONSIDERO TEMP9AM, SEMBRA RAGIONEVOLE PENSARE CHE SI AVVICINI A UNA NORMALE
q = quantile(dataset$Temp9am)
hist(dataset$Temp9am, main = "Distribuzione Temp9am", xlab = "Temp9am")
abline(v = q[1], col = "red", lwd = 2) # 0% (min)
abline(v = q[2], col = "blue", lwd = 2) # 1st quartile 25%
abline(v = q[3], col = "green", lwd = 2.5) # Median value distribution 50%
abline(v = q[4], col = "blue", lwd = 2) # 3rd quartile 75%
abline(v = q[5], col = "red", lwd = 2) # 100% (max)
boxplot(dataset$Temp9am, main = "Distribuzione Temp9am", xlab = "Temp9am")
hist(dataset$Temp9am, main = "Distribuzione Temp9am", xlab = "Temp9am", freq = FALSE)
lines(density(dataset$Temp9am), col = "red")
#CONTROLLO CHE L'ASSUNZIONE SIA VALIDA E OSSERVO IL VALORE MINIMO, MASSIMO, MEDIA E DEVIAZIONE STANDARD
descdist(dataset$Temp9am, discrete = FALSE)
#SEMBRA CHE UNA DISTRIBUZIONE APPROPRIATA SIA LA NORMALE
fit.norm <- fitdist(dataset$Temp9am, "norm")
plot(fit.norm)

#CONSIDERO TEMP3PM, SEMBRA RAGIONEVOLE PENSARE CHE SI AVVICINI A UNA NORMALE
q = quantile(dataset$Temp3pm)
hist(dataset$Temp3pm, main = "Distribuzione Temp3pm", xlab = "Temp3pm")
abline(v = q[1], col = "red", lwd = 2) # 0% (min)
abline(v = q[2], col = "blue", lwd = 2) # 1st quartile 25%
abline(v = q[3], col = "green", lwd = 2.5) # Median value distribution 50%
abline(v = q[4], col = "blue", lwd = 2) # 3rd quartile 75%
abline(v = q[5], col = "red", lwd = 2) # 100% (max)
boxplot(dataset$Temp3pm, main = "Distribuzione Temp3pm", xlab = "Temp3pm")
hist(dataset$Temp3pm, main = "Distribuzione Temp3pm", xlab = "Temp3pm", freq = FALSE)
lines(density(dataset$Temp3pm), col = "red")
#CONTROLLO CHE L'ASSUNZIONE SIA VALIDA E OSSERVO IL VALORE MINIMO, MASSIMO, MEDIA E DEVIAZIONE STANDARD
descdist(dataset$Temp3pm, discrete = FALSE)
#SEMBRA CHE UNA DISTRIBUZIONE APPROPRIATA SIA LA NORMALE
fit.norm <- fitdist(dataset$Temp3pm, "norm")
plot(fit.norm)

#CONSIDERO RAINTODAY: SI HA UNO SBILANCIAMENTO SIGNIFICATIVO TRA 0 (77,5) E 1 (22,5%)
table(dataset$RainToday)
plot(dataset$RainToday, main = "Distribuzione RainToday", xlab = "RainToday", col = c("black", "white"))

#CONSIDERO I VALORI DEL TARGET RAINTOMORROW: SI HA UNO SBILANCIAMENTO SIGNIFICATIVO TRA 0 (77,8) E 1 (22,2%)
table(dataset$RainTomorrow)
plot(dataset$RainTomorrow, main = "Distribuzione RainTomorrow", xlab = "RainTomorrow", col = c("black", "white"))

#MATRICE DI CORRELAZIONE 
#CONSIDERO SOLO GLI ATTRIBUTI NUMERICI PER LA CORRELAZIONE

M <- cor(dataset[-c(4, 6:7, 16:17)])
M

#brewer.pal(n=8, name="RdBu")
#method = color, number o circle
col <- colorRampPalette(c("#BB4444", "#EE9988", "#FFFFFF", "#77AADD", "#4477AA"))
corrplot(M, method="color", col=col(200),
         type="upper", order="hclust", 
         addCoef.col="black", # AGGIUNGE COEFFICENTE DI CORRELAZIONE
         tl.col="black", tl.srt=45, # COLORE E ROTAZIONE DELLA LABEL
         diag=F # NASCONDE LA CORRELAZIONE NELLA DIAGONALE PRINCIPALE (TUTTA = A 1)
)

#FEATURE SELECTION:
#CALCOLO ATTRAVERSO LA FUNZIONE FINDCORRELATION QUALI SONO GLI ATTRIBUTI DA RIMUOVERE IN BASE ALLA LORO CORRELAZIONE
#MOTIVO: SE DUE ATTRIBUTI HANNO CORRELAZIONE MOLTO ALTA POSSO NON CONSIDERARNE UNO DEI DUE
#COME FUNZIONA: SE DUE VARIABILI HANNO ALTA CORRELAZIONE, LA FUNZIONE CONTROLLA LA CORRELAZIONE ASSOLUTA MEDIA DI 
#OGNI VARIABILE E RIMUOVE QUELLA CON VALORE MAGGIORE
highlyCorrelated <- findCorrelation(M, cutoff = 0.67, names = TRUE, verbose = TRUE)
#VISUALIZZO I NOMI DEGLI ATTRIBUTI DA RIMUOVERE
highlyCorrelated

#FACCIO UNA COPIA DEL DATASET, PRIMA DI RIMUOVERE GLI ATTRIBUTI, PER POTER EFFETTUARE PCA
datasetPCA <- dataset
#RIMUOVO GLI ATTRIBUTI ("Temp9am", "MaxTemp", "Temp3pm", "Pressure3pm", "Humidity9am", "WindGustSpeed")
dataset <- dataset[-c(2, 5, 10, 13:15)]

#VISUALIZZO LE COLONNE DEL DATASET RIDOTTO
colnames(dataset)

#PCA (VOGLIO AVERE UNA CONFFERMA DEI RISULTATI DERIVANTI DURANTE L'ANALISI DI CORRELAZIONE)
#CALCOLO PCA
res.pca <- PCA(datasetPCA[-c(4, 6:7, 16:17)], scale.unit = TRUE, graph = FALSE)

#OTTENGO E MOSTRO GLI AUTOVALORI
eig.val <- get_eigenvalue(res.pca)
eig.val
#VALORI >1 MI INDICANO CHE LA DIMENSIONE CONSIDERATA HA IMPORTANZA E DEVE ESSERE CONSIDERATA (NEL NOSTRO CASO QUINDI
#CONSIDERIAMO LE PRIME 4 DIMENSIONI)

#ALTRIMENTI POSSIAMO ATTRAVERSO IL GRAFICO SEGUENTE CONSIDERARE DI ARRIVARE AD UNA CERTA "SOGLIA" DI QUANTO IL DATASET
#VIENE DESCRITTO
fviz_eig(res.pca, addlabels = TRUE, ylim = c(0, 50), title = "PCA")
#PER ESEMPIO SE ANCORA UNA VOLTA CONSIDERASSI SOLO LE PRIME 4 DIMENSIONI QUESTE MI DESCRIVEREBBERO L'82% DEL DATASET

#OTTENGO INFORMAZIONE DAL PCA SULLE VARIABILI
var <- get_pca_var(res.pca)
#IN PARTICOLARE MI POTREBBERO INTERESSARE LA CORRELAZIONE TRA LE VARIABILI E LE DIMENSIONI O IL LORO CONTRIBUTO
var$cor
var$contrib
#INOLTRE POSSO VEDERE QUANTO LE VARIABILI SONO CORRELATE TRA DI LORO ATTRAVERSO QUESTO GRAFICO: TRAMITE VISUALIZZAZIONE
#E' FACILE CONFERMARE CIO' CHE E' EMERSO ATTRAVERSO LA MATRICE DI CORRELAZIONE
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

plot(trainset$RainTomorrow, col=c("red", "green"), names = c("No", "Yes"))
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
dummy.pred = 0
dummy.pred = factor(dummy.pred)

#UNA  VOLTA AGGIUNTA LA COLONNA DELLE PREVISIONI AL TRAINSET MISURO LE PERFORMANCE (CONF MATRIX E ACCURACY)
confMatrix <- table(dummy.pred, testset$RainTomorrow)
confMatrix
print(paste("Accuracy: ", sum(diag(confMatrix))/sum(confMatrix)))

#OPPURE
confusionMatrix(dummy.pred, testset$RainTomorrow, mode ="everything")

#UN ACCURACY DI 77.6%, DECISAMENTE NON MALE VISTO CHE NON ABBIAMO INTRDOTTO NESSUN MODELLO E NON ABBIAMO UTILIZZATO
#RISORSE. 
#N.B. QUESTO RISULATATO COSI' BUONO E' IL FRUTTO DELLA VARIABILE TARGET SBILANCIATA!!!!!
#IL NOSTRO OBIETTIVO SARA' QUINDI QUELLO DI CREARE DEI MODELLI PREDITTIVI CHE MIGLIORINO IL 77% DI QUESTO PRIMO APPROCCIO BASE

#COME PROVA DELLE ANALISI FATTE IN PRECEDENZA POSSIAMO VEDERE QUALI SONO LE PERFORMANCE SE AL POSTO DI TUTTI 0
#IMPONGO CHE SE IN UN DATO GIORNO PIOVE ALLORA PREVEDO CHE CI SARA' PIOGGIA ANCHE IN QUELLO SUCCESSIVO
#REMINDER: ERA STATO DETTO CHE QUESTA ASSUNZIONE NON E' VERIFICATA...
dummy.pred[testset$RainToday == 1] = 1
dummy.pred = factor(dummy.pred)

confusionMatrix(dummy.pred, testset$RainTomorrow)
#TALE ASSUNZIONE RISULTA ANCORA UNA VOLTA NON FONDATA IN QUANTO LE PERFORMANCE DEL MODELLO SONO DIMINUITE RISPETTO
#AL PRIMO APPROCCIO (CIOE' QUELLO CON TUTTI 0) (DA 77.6% A 76.4%)


#MODELLO 1: DECISION TREE / RANDOM FOREST

#PER PRIMA COSA COSTRUISCO UN ALBERO DI DECISIONE LIBERO DI CRESCERE LIBERAMENTE
bigDecisionTree = rpart(RainTomorrow ~ ., 
                        data = trainset, method = "class", control = rpart.control(cp = 0))

#VISUALIZZO GRAFICAMENTE IL PLOT DELL'ALBERO E DELLA SUA COMPLESSITA' DEI PARAMETRI
plot(bigDecisionTree)
plotcp(bigDecisionTree)
#MI ACCORGO SUBITO CHE OLTRE UN CERTO PUNTO OLTRE CHE DIVENTARE SEMPRE PIU' ONEROSO IN TERMINI COMPUTAZIONALI 
#L'ALBERO NON MIGLIORA PIU' DI UN CERTO LIMIRE

#CALCOLO LA PREDICTION UTILIZZANDO QUESTO MODELLO
bigdt.pred <- predict(bigDecisionTree, testset, type = "class")

#CALCOLO LE PERFORMANCE DEL MODELLO
confMatrix = table(bigdt.pred, testset$RainTomorrow)
confMatrix
print(paste("Accuracy Big Decision Tree: ", sum(diag(confMatrix))/sum(confMatrix)))

confusionMatrix(bigdt.pred, testset$RainTomorrow, mode ="everything")
#VEDO CHE COMUNQUE L'ACCURACY E' MIGLIORE RISPETTO AL PRIMO MODELLO DUMMY (DA 77.6% A 81% CIRCA), DEVO CERCARE PERO' DI OTTIMIZZARE IL MIO ALBERO

#VOGLIO CAPIRE COME DI COMPORTA SE LIMITO AD UN VALORE RAGIONEVOLE LA PROFONDITA' MASSIMA DI CRESCITA DELL'ALBERO DI DECISIONE
decisionTree = rpart(RainTomorrow ~ ., 
                     data = trainset, method = "class", control = rpart.control(cp = 0, maxdepth = 6)) 
#HO CREATO IL MODELLO ALLENATO SUL TRAINSET

#VISUALIZZO GRAFICAMENTE IL PLOT DELL'ALBERO E DELLA SUA COMPLESSITA' DEI PARAMETRI
plot(decisionTree)
plotcp(decisionTree)
#NOTO GIA' "A OCCHIO" CHE LA COMPLESSITA' DELL'ALBERO E' DIMINUITA

#CALCOLO LA PREDICTION UTILIZZANDO QUESTO MODELLO
dt.pred <- predict(decisionTree, testset, type = "class")

#CALCOLO LE PERFORMANCE DEL MODELLO
confMatrix = table(dt.pred, testset$RainTomorrow)
confMatrix
print(paste("Accuracy Decision Tree: ", sum(diag(confMatrix))/sum(confMatrix)))
#DALLE PERFORMANCE VEDO ANCHE CHE L'ACCURACY E' MIGLIORATA PASSANDO DA 81% CIRCA A 84%
confusionMatrix(dt.pred, testset$RainTomorrow, mode ="everything")

#MI ACCORGO PERO' CHE POTREI RENDERE IL MODELLO ANCORA PIU' "SNELLO" E COMPUTAZIONALMENTE PIU' EFFICIENTE
#UTILIZZO LA FUNZIONE "prune" PER POTARE L'ALBERO, QUESTA EVITA IL FENOMENO DELL'OVERFITTING E RIDUCE LA COMPLESSITA'
#PER SCEGLIERE UN CP OPPORTUNO GUARDO IL GRAFICO PLOTCP DEL DECISION TREE CHE VOGLIO POTARE
prunedDecisionTree = prune(decisionTree, cp = 0.017)

#CREO UN ALBERO POTATO

#ESSENDO MOLTO PIU' RIDOTTO POSSO VISUALIZZARLO IN MODO MIGLIORE
fancyRpartPlot(prunedDecisionTree, sub = "")
#VEDO CHE IL MODELLO CREATO UTILIZZA AL MASSIMO 3 ATTRIBUTI PER ASSOCIARE UN ETICHETTA ALLE OSSERVAZIONI,
#VOGLIO PERO' CONTROLLARE L'ACCURACY CHE MI GARANTISCE QUESTO MODELLO "SEMPLIFICATO"

#CALCOLO LA PREDICTION UTILIZZANDO QUESTO MODELLO SEMPLIFICATO
pdt.pred <- predict(prunedDecisionTree, testset, type = "class")

#CALCOLO LE PERFORMANCE DEL MODELLO
confMatrix = table(pdt.pred, testset$RainTomorrow)
confMatrix
print(paste("Accuracy Pruned Decision Tree: ", sum(diag(confMatrix))/sum(confMatrix)))

confusionMatrix(pdt.pred, testset$RainTomorrow, mode ="everything")
#HO UNA ACCURACY DEL 83.5% CIRCA E' LEGGERMENTE MINORE RISPETTO AL MODELLO PRECEDENTE (84%) MA MOLTO MIGLIORE IN TERMINI 
#DI COMPLESSITA' E QUINDI E' IL MODELLO DA PREFERIRE!!

#ALLENO ORA UNA RANDOM FOREST
randomForest = randomForest(RainTomorrow ~ ., 
                            data = trainset, method = "class", ntree = 500)

#TALE MODELLO E' DECISAMENTE PIU' LENTO DI DECISION TREE PERCHE' ALLENA DIVERSI ALBERI DECISIONALI PER POI TENERE IL MIGLIORE
#TEMPO: 2 MINUTI

#CREO LA PREVISIONE UTILIZZANDO IL MODELLO ALLENATO
randf.pred <- predict(randomForest, testset, type = "class")

#CALCOLO LE PERFORMANCE DEL MODELLO
confMatrix = table(randf.pred, testset$RainTomorrow)
confMatrix
print(paste("Accuracy Random Forest: ", sum(diag(confMatrix))/sum(confMatrix)))

confusionMatrix(randf.pred, testset$RainTomorrow, mode ="everything")
#SI PUO' NOTARE PERO' CHE IL MODELLO E' PIU' PRECISO ARRIVANDO A SFIORARE L'85% DI ACCURACY


#MODELLO 2: NAIVE BAYES
naiveBayes = naiveBayes(RainTomorrow ~ ., data = trainset, type = "class")

#CREO LA PREVISIONE UTILIZZANDO IL MODELLO ALLENATO
nb.pred <- predict(naiveBayes, testset, type = "class")

#CALCOLO LE PERFORMANCE DEL MODELLO
confMatrix = table(nb.pred, testset$RainTomorrow)
confMatrix
print(paste("Accuracy Naive Bayes: ", sum(diag(confMatrix))/sum(confMatrix)))
#NAIVE BAYES MI DA UN ACCURACY DEL 82%
confusionMatrix(nb.pred, testset$RainTomorrow, mode = "everything")


#MODELLO 3: NEURAL NETWORK
levels(trainset$RainToday) <- c("No", "Yes")
levels(trainset$RainTomorrow) <- c("No", "Yes")
levels(testset$RainToday) <- c("No", "Yes")
levels(testset$RainTomorrow) <- c("No", "Yes")

nn= train(RainTomorrow ~ ., data=trainset, method = "nnet", trControl = trainControl(method = "cv", number = 1,
                                classProbs = TRUE, summaryFunction = twoClassSummary, verboseIter = TRUE))

#CREO LA PREVISIONE UTILIZZANDO IL MODELLO ALLENATO
neunet.probs = predict(nn, testset[,! names(testset) %in% c("RainTomorrow")], type = "prob")
neunet.pred = ifelse(neunet.probs$No > neunet.probs$Yes, "No", "Yes") 
neunet.pred = factor(neunet.pred)

#CALCOLO LE PERFORMANCE DEL MODELLO
confMatrix = table(neunet.pred, testset$RainTomorrow)
confMatrix
print(paste("Accuracy Neural Network: ", sum(diag(confMatrix))/sum(confMatrix)))

confusionMatrix(neunet.pred, testset$RainTomorrow, mode ="everything", positive = "Yes")

levels(trainset$RainToday) <- c("0", "1")
levels(trainset$RainTomorrow) <- c("0", "1")
levels(testset$RainToday) <- c("0", "1")
levels(testset$RainTomorrow) <- c("0", "1")

#MODELLO 4: SVM
svm = svm(RainTomorrow ~ ., data = trainset, kernel = 'radial', scale = TRUE) #linear, polynomial, radial, sigmoid

#CREO LA PREVISIONE UTILIZZANDO IL MODELLO ALLENATO
svm.pred <- predict(svm, testset, type = "class")

#CALCOLO LE PERFORMANCE DEL MODELLO
confMatrix = table(svm.pred, testset$RainTomorrow)
confMatrix
print(paste("Accuracy SVM: ", sum(diag(confMatrix))/sum(confMatrix)))

confusionMatrix(svm.pred, testset$RainTomorrow, mode ="everything")
#CON UN PRIMO APPROCCIO IL MODELLO SVM HA UNA ACCURACY DEL 84% CIRCA IMPIEGANDOCI PERO' MOLTO PIU' TEMPO DEI MODELLI
#PRECEDENTI (CIRCA 15 MINUTI PER ALLENARE IL MODELLO)

#PROVIAMO CON DATASET SENZA FATTORI
trainsetSVM = trainset[-c(3:5)]
testsetSVM = testset[-c(3:5)]

svm = svm(RainTomorrow ~ ., data = trainsetSVM, kernel = 'sigmoid', scale = TRUE)

#CREO LA PREVISIONE UTILIZZANDO IL MODELLO ALLENATO
svm.pred <- predict(svm, testsetSVM, type = "class")

#CALCOLO LE PERFORMANCE DEL MODELLO
confMatrix = table(svm.pred, testsetSVM$RainTomorrow)
confMatrix
print(paste("Accuracy SVM: ", sum(diag(confMatrix))/sum(confMatrix)))

confusionMatrix(svm.pred, testset$RainTomorrow, mode ="everything")

#PERFORMANCE KERNERL CON DATASET COMPLETO: 
#1)LINEAR: 14 MINUTI - 84% ACCURACY
#2)POLYNOMIAL: 8 MINUTI - 82.5% ACCURACY
#3)RADIAL: 15 MINUTI - 84% ACCURACY
#4)SIGMOID: 6 MINUTI - 79% ACCURACY

#PERFORMANCE KERNERL CON DATASET SENZA FATTORI:
#1)LINEAR: 7 MINUTI - 84% ACCURACY
#2)POLYNOMIAL:  12 MINUTI - 84% ACCURACY
#3)RADIAL: 17 MINUTI - 84% ACCURACY
#4)SIGMOID: 8 MINUTI - 76% ACCURACY

#-------- esperimenti 10 fold cross validation & ROC --------

#IN TUTTI I MODELLI SEGUENTI UTILIZZEREMO LA 10 FOLD CROSS VALIDATION PER ...

#METTO A CONFRONTO I DUE MODELLI CHE HANNO DATO RISULTATI MIGLIORI APPLICANDO UNA 10 FOLD CROSS VALIDATION
#E CONTROLLANDO LE PERFORMANCE TRAMITE ROC E AUC

#CONTROLLO CHE ESEGUE UNA 10 FOLD CROSS VALIDATION IN OGNI MODELLO
control = trainControl(method = "cv", number = 10,
                    classProbs = TRUE, summaryFunction = twoClassSummary, verboseIter = TRUE)

levels(trainset$RainToday) <- c("No", "Yes")
levels(trainset$RainTomorrow) <- c("No", "Yes")
levels(testset$RainToday) <- c("No", "Yes")
levels(testset$RainTomorrow) <- c("No", "Yes")

#CREAZIONE MODELLI CON 10 FOLD CROSS VALIDATION

#DECISION TREE
rpart.model= train(RainTomorrow ~ ., data=trainset, method = "rpart2", metric = 
                  "ROC", trControl = control, tuneGrid = expand.grid(.maxdepth = 2:10))

#VISUALIZZO PRIME 10 VARIABILI PER IMPORTANZA PER RPART
imp = varImp(rpart.model)
plot(imp, top = 10)

#RANDOM FOREST
rf.model= train(RainTomorrow ~ ., data=trainset, method = "rf", metric = 
                  "ROC", trControl = control)

#VISUALIZZO PRIME 10 VARIABILI PER IMPORTANZA PER RANDOM FOREST
imp = varImp(rf.model)
plot(imp, top = 10)

#NAIVE BAYES
naiveBayes.model= train(RainTomorrow ~ ., data=trainset, method = "naive_bayes", metric = 
                          "ROC", trControl = control)

#VISUALIZZO PRIME 10 VARIABILI PER IMPORTANZA PER NAIVE BAYES
imp = varImp(naiveBayes.model)
plot(imp, top = 10)


#NEURAL NETWORK
nn.model= train(RainTomorrow ~ ., data=trainset, method = "nnet", metric = 
                  "ROC", trControl = control)

#VISUALIZZO PRIME 10 VARIABILI PER IMPORTANZA PER NEURAL NETWORK
imp = varImp(nn.model)
plot(imp, top = 10)

#CALCOLO PROBABILITY
rpart.probs = predict(rpart.model, testset[,! names(testset) %in% c("RainTomorrow")],
                      type = "prob")
naiveBayes.probs = predict(naiveBayes.model, testset[,! names(testset) %in% c("RainTomorrow")],
                           type = "prob")
rf.probs = predict(rf.model, testset[,! names(testset) %in% c("RainTomorrow")],
                   type = "prob")
nn.probs = predict(nn.model, testset[,! names(testset) %in% c("RainTomorrow")],
                   type = "prob")

#CALCOLO ROC
rpart.ROC = roc(response = testset[,c("RainTomorrow")], predictor = rpart.probs$Yes,
                levels = levels(testset[,c("RainTomorrow")]))

naiveBayes.ROC = roc(response = testset[,c("RainTomorrow")], predictor = naiveBayes.probs$Yes,
                levels = levels(testset[,c("RainTomorrow")]))

rf.ROC = roc(response = testset[,c("RainTomorrow")], predictor = rf.probs$Yes,
                     levels = levels(testset[,c("RainTomorrow")]))

nn.ROC = roc(response = testset[,c("RainTomorrow")], predictor = nn.probs$Yes,
                     levels = levels(testset[,c("RainTomorrow")]))

#PLOT ROC CURVES
plot(rpart.ROC, col="blue")
plot(naiveBayes.ROC, col="green", add=T)
plot(rf.ROC, col="black", add=T)
plot(nn.ROC, col="orange", add=T)

abline(v=c(0,1))
legend("topright", legend = c("rpart", "naive bayes", "random forest", "neural network"), 
        col=c("blue", "green", "black", "orange"), cex = 0.3, lty = 1)

#STATISTICHE ROC MODELLI
rpart.ROC
naiveBayes.ROC
rf.ROC
nn.ROC

#CALCOLO PREDICTION DALLE PROBABILITY E PERFORMANCE
rpart.pred = ifelse(rpart.probs$No > rpart.probs$Yes, "No", "Yes")
rpart.pred = factor(rpart.pred)
rf.pred = ifelse(rf.probs$No > rf.probs$Yes, "No", "Yes") 
rf.pred = factor(rf.pred)
naiveBayes.pred = ifelse(naiveBayes.probs$No > naiveBayes.probs$Yes, "No", "Yes") 
naiveBayes.pred = factor(naiveBayes.pred)
nn.pred = ifelse(nn.probs$No > nn.probs$Yes, "No", "Yes") 
nn.pred = factor(nn.pred)

confusionMatrix(rpart.pred, testset$RainTomorrow,  mode = "everything")
confusionMatrix(naiveBayes.pred, testset$RainTomorrow,  mode = "everything")
confusionMatrix(rf.pred, testset$RainTomorrow,  mode = "everything")
confusionMatrix(nn.pred, testset$RainTomorrow,  mode = "everything")
