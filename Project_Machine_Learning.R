library(pastecs)
library(ggplot2)
library(dplyr)
library(tidyr) 
library(tidyverse)
library(stargazer)
library(plyr)
library(caret)
library(plotly)
library(dplyr)
library(polycor)
library(rgeos)
library(tensorflow)
library(corrplot) 
library(ggpubr)
library(rgeos)
library(gtools)
library(lubridate)
library(xts)

library(zoo)
library(quantmod)
library(readr)
library(repr)
reticulate::use_condaenv("tf1")
library(tensorflow)
library(keras)
library(devtools)
library(data.table)
library(TTR)
library(forecast)
library(reticulate)

options(scipen=999)
options(digits=3)

setwd('C:/Users/Cilen/OneDrive/Documenti/Progetto machine learning')
consegne.vaccini.latest <- read.csv("consegne-vaccini-latest.csv")
dd=consegne.vaccini.latest

summary(dd)
str(dd)
### sostituisco valori negativi con zero
# dati$numero_dosi[dati$numero_dosi<0]<-0

#### rimuovo dati con valori neagativi
dd2<-dd[dd$numero_dosi>=0, ]

drop<-c("area","codice_NUTS1","codice_NUTS2","codice_regione_ISTAT","nome_area" )
dd3<-dd2[,!(names(dd2) %in% drop)]

dati1<-dd3 %>% group_by(fornitore, data_consegna) %>%
  summarise_all(sum)

dati1$soglia<-ifelse(dati1$numero_dosi>=1000, 1,0)

dati1$date=ymd(dati1$data_consegna)
ddate=dati1$date
str(dati1)
(dati1)
#### questa deve girare solo una volta (se per sbaglio lo fai girare
### più vilte, torna  riga 43 e rigenera dati1
#dati1<-dati1[,-2]
##########################
##### separa per fornitore
d_pfizer<-dati1[which(dati1$fornitore=="Pfizer/BioNTech"),]
date_pfizer=ymd(d_pfizer$date)
pfizer_ts<-as.xts(d_pfizer[,2:3], order.by = date_pfizer)

d_JNJ<-dati1[which(dati1$fornitore=="Janssen"),]
date_JNJ=ymd(d_JNJ$date)
jnj_ts<-as.xts(d_JNJ[,2:3], order.by = date_JNJ)

d_ASTRZ<-dati1[which(dati1$fornitore=="Vaxzevria (AstraZeneca)"),]
date_Astrz=ymd(d_ASTRZ$date)
astrz_ts<-as.xts(d_ASTRZ[,2:3], order.by = date_Astrz)

d_Moderna<-as.data.frame(dati1[which(dati1$fornitore=="Moderna"),])
str(d_Moderna)
date_moderna=ymd(d_Moderna$date)
moderna_ts<-xts(d_Moderna[,2:3], order.by = date_moderna)
str(moderna_ts)
plot(moderna_ts$numero_dosi)


write.csv(d_Moderna,"Moderna.csv")
write.csv(d_ASTRZ,"ASTRZ.csv")
write.csv(d_JNJ,"JNJ.csv")
write.csv(d_pfizer,"pfizer.csv")


#######
d_final_date<-index(moderna_ts)
d_final<-as.data.frame(moderna_ts)
d_final$date<-d_final_date
data=as.zoo(d_final)

data2=d_final$soglia

length(data2)
#data2[:220]
#### Ricorda di mettere uguali 
n_steps_in=15
n_steps_out=5
n_tot=n_steps_in+n_steps_out

### ciclo funzione oper creare vettori x, y
X=matrix(0,nrow=length(data2)-n_tot, ncol=n_steps_in)
y=matrix(0,nrow=length(data2)-n_tot, ncol=n_steps_out)
X=list()
y=list()
for(i in 1:length(data2)){
  end_ix=i+n_steps_in-1
  out_end_ix=end_ix+n_steps_out-1
  seq_x=data2[i:end_ix]
  seq_y=data2[end_ix:out_end_ix]
  if(out_end_ix>length(data2)){
    break
  }
  print(seq_y)
  
  X[[i]]=seq_x
  y[[i]]=seq_y
}


### converto liste in array che possono essere messe in rete neurale
XX=(matrix(unlist(X),nrow=length(X),ncol=n_steps_in, byrow=TRUE))
XX=array(data = XX,dim = c(length(X),n_steps_in))

yy=(matrix(unlist(y),nrow=length(y),ncol=n_steps_out, byrow=TRUE))
yy=array(data = yy,dim = c(length(y),n_steps_out))


n_input=ncol(XX)
n_output=ncol(yy)

### scrivo rete
model <- keras_model_sequential()
model%>%
  layer_dense(units = 16, activation = 'sigmoid', input_shape =c(n_input))%>% 
  layer_dense(units = n_output) %>% 
  compile(
    optimizer =optimizer_sgd(),
    loss = 'binary_crossentropy',
    metrics = c('accuracy')
  )

#### adatto modello ai dati
model %>% fit(XX, yy, epochs=50, batch_size=1,  shuffle = FALSE,
              validation_split=0.2)

### genero una previsione 
y_hat <- model %>%
  predict(XX, batch_size=1) 


#### confusion Matrix 
yy_conf=factor(as.vector(yy))

y_hat_conf=as.vector(y_hat)
### valori sopra 0.5 diventano 1; sotto diventano 0
y_hat_conf<-factor(ifelse(y_hat_conf>=0.5,1,0))

confusionMatrix(data=yy_conf, reference = y_hat_conf)

