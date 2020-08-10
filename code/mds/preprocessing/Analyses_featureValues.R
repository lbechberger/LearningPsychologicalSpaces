# Das ist ein Orientierungs-Skript für die Auswertung der Feature value Daten (both feature analyses and control analysis zu features)
# Namen von Spalten & Co müssen an den konkreten Datensatz angepasst werden.
#-----------------
# CONTENT of the Skript
# für att. feature analysis:          Korrelation zwischen den 3 Merkmalen?                         - ab line 38
# für pre-att. feature analysis:      Korrelation zwischen den 3 Merkmalen?                         - ab line 78
# for control analysis (features):    Korrelieren att und pre-att (feature-based)?                  - ab line 115
# für control analysis (features):    Unterschied att vs preatt results (Einfluss der PerceptDur?)  - ab line 153

#-----------------
# PACKAGES (ggf. install.packages("xy") vorher)
library(lme4)           # für Linear Mixed Models
library(lmerTest)       # für p-Werte zu LMMs
library(car)            # für qqPlot() beim Testen der assumptions
library(ggplot2)        # für Plots - falls man die in R erstellen will
#-----------------
# Korrelationsanalysen: benötigtes csv-FILE (enthält am besten keine NAs, ansonsten muss das in den späteren Funktionen berücksichtigt werden):
# Data file sollte die folgenden Spalten mind. haben (für jedes Bild eine Zeile! - das ist anders als für MixedModels):
# Pic:        eine Spalte, wo kodiert ist, um welches Bild es geht (factor mit 60 levels)
# att_Form:     eine Spalte, wo der Bild-mean aus den attentive Form-data drinsteht (numeric variable)
# att_Orie:     eine Spalte, wo der Bild-mean aus den attentive Orientation-data drinsteht (numeric variable)
# att_Line:     eine Spalte, wo der Bild-mean aus den attentive Lines-data drinsteht (numeric variable)
# preatt_Form:     eine Spalte, wo der Bild-mean aus den pre-attentive Form-data drinsteht (numeric variable)
# preatt_Orie:     eine Spalte, wo der Bild-mean aus den pre-attentive Orientation-data drinsteht (numeric variable)
# preatt_Line:     eine Spalte, wo der Bild-mean aus den pre-attentive Lines-data drinsteht (numeric variable)

#-----------------
# Daten einlesen mit passender Funktion, z.B. 
dat<- read.csv2()

# Kodierung der Spalten checken, z.B.:
str(dat)

# > all 6 Spalten wo die Bild-means drin stehen müssen numeric sein, falls nicht dann umcodieren, z.B. so:
dat$att_Form <- as.numeric(dat$att_Form)

#-----------------
# Pearson-Korelation für att. feature analysis:  Korrelation zwischen den 3 Merkmalen? 

#-----------------
# cor.test() gehört zum base system von R
# kann für Pearson + Spearman-Correlations genutzt werden
# gibt: r-Werte, p-Werte und CIs
# wir rechnen Pearson (weil mind. intervallskaliert), two-tailed (weil wir keine speziellen Hypothesen haben), und standard-CI von 95%
#-----------------

#__________Assumptions checken
# alle samples (daten aus den 3 numeric att_xxxx-Spalten) müssen jeweils annähernd normal verteilt sein (reicht visual inspection, denke ich)

#__________FORM vs. ORIENTATION

cor.test(dat$att_Form, dat$att_Orie, method="pearson")

# > folgende Daten brauchen wir dann (auf 2 Nachkommastellen gerundet):
# r (corr aus sample estimates)
# CI(x,y) (die 2 Werte unten 95 percent confidence interval)
# t(df)=.., p< .. (die Levels of interest sind: p<.05, <.01. <.001 und <.0001)

#__________FORM vs. LINES

cor.test(dat$att_Form, dat$att_Line, method="pearson")

# > folgende Daten brauchen wir dann (auf 2 Nachkommastellen gerundet):
# r (corr aus sample estimates)
# CI(x,y) (die 2 Werte unten 95 percent confidence interval)
# t(df)=.., p< .. (die Levels of interest sind: p<.05, <.01. <.001 und <.0001)

#__________LINES vs. ORIENTATION

cor.test(dat$att_Line, dat$att_Orie, method="pearson")

# > folgende Daten brauchen wir dann (auf 2 Nachkommastellen gerundet):
# r (corr aus sample estimates)
# CI(x,y) (die 2 Werte unten 95 percent confidence interval)
# t(df)=.., p< .. (die Levels of interest sind: p<.05, <.01. <.001 und <.0001)

#-----------------
# Pearson-Korelation für pre-att. feature analysis:  Korrelation zwischen den 3 Merkmalen? 

#-----------------
# wir rechnen Pearson (weil mind. intervallskaliert),  two-tailed (weil wir keine speziellen Hypothesen haben), und standard-CI von 95%
#-----------------

#__________Assumptions checken
# alle samples (daten aus den 3 numeric preatt_xxxx-Spalten) müssen jeweils annähernd normal verteilt sein (reicht visual inspection, denke ich)

#__________FORM vs. ORIENTATION

cor.test(dat$preatt_Form, dat$preatt_Orie, method="pearson")

# > folgende Daten brauchen wir dann (auf 2 Nachkommastellen gerundet):
# r (corr aus sample estimates)
# CI(x,y) (die 2 Werte unten 95 percent confidence interval)
# t(df)=.., p< .. (die Levels of interest sind:p <.05, <.01. <.001 und <.0001)

#__________FORM vs. LINES

cor.test(dat$preatt_Form, dat$preatt_Line, method="pearson")

# > folgende Daten brauchen wir dann (auf 2 Nachkommastellen gerundet):
# r (corr aus sample estimates)
# CI(x,y) (die 2 Werte unten 95 percent confidence interval)
# t(df)=.., p< .. (die Levels of interest sind: p<.05, <.01. <.001 und <.0001)

#__________LINES vs. ORIENTATION

cor.test(dat$preatt_Line, dat$preatt_Orie, method="pearson")

# > folgende Daten brauchen wir dann (auf 2 Nachkommastellen gerundet):
# r (corr aus sample estimates)
# CI(x,y) (die 2 Werte unten 95 percent confidence interval)
# t(df)=.., p< .. (die Levels of interest sind: p<.05, <.01. <.001 und <.0001)

#-----------------
# Pearson-Korelation für control analysis zu features:  Korrelation zwischen att und preatt results? 

#-----------------
# wir rechnen für jedes Feature separat
# wir rechnen Pearson (weil mind. intervallskalierte Daten), two-tailed (weil wir keine speziellen Hypothesen haben), und standard-CI von 95%
#-----------------

#__________Assumptions checken
# normality ist bereits gecheckt bei den Einzelanalysen, muss für den hier ausgewerteten Daten ebenfalls gelten (ggf. Modifikationen übernehmen)

#__________FORM att vs. preatt

cor.test(dat$att_Form, dat$preatt_Form, method="pearson")

# > folgende Daten brauchen wir dann (auf 2 Nachkommastellen gerundet):
# r (corr aus sample estimates)
# CI(x,y) (die 2 Werte unten 95 percent confidence interval)
# t(df)=.., p< .. (die Levels of interest sind: p<.05, <.01. <.001 und <.0001)

#__________ORIENTATION att vs. preatt

cor.test(dat$att_Orie, dat$preatt_Orie, method="pearson")

# > folgende Daten brauchen wir dann (auf 2 Nachkommastellen gerundet):
# r (corr aus sample estimates)
# CI(x,y) (die 2 Werte unten 95 percent confidence interval)
# t(df)=.., p< .. (die Levels of interest sind: p<.05, <.01. <.001 und <.0001)

#__________LINES att vs. preatt

cor.test(dat$att_Line, dat$preatt_Line, method="pearson")

# > folgende Daten brauchen wir dann (auf 2 Nachkommastellen gerundet):
# r (corr aus sample estimates)
# CI(x,y) (die 2 Werte unten 95 percent confidence interval)
# t(df)=.., p< .. (die Levels of interest sind: p<.05, <.01. <.001 und <.0001)

#-----------------
#-----------------
# Mixed Models: benötigtes csv-FILE (enthält am besten keine NAs, ansonsten muss das in den späteren Funktionen berücksichtigt werden):
# Data file sollte die folgenden Spalten mind. haben (für jeden gemessenen Wert eine Zeile! - das ist anders als bei den Korrelationsanalysen!):
# Pic:          eine Spalte, wo kodiert ist, um welches Bild es geht (factor mit 60 levels)
# ratingType:   eine Spalte, wo kodiert ist, ob Wert aus att-Studie oder pre-att-Studie (factor mit 2 levels)
# feature:      eine Spalte, wo kodiert ist, welches Feature abgefragt wurde (z.B. factor mit 3 levels oder anders)
# values:       eine Spalte, wo der gemessene Wert drin steht (numeric variable)
# (SubjID:      eine Spalte, wo kodiert ist, von welchem Probanden der Wert stammt (factor mit 89? levels)) 
# (visualType:  eine Spalte, wo kodiert ist, ob das Objekt zu einer coherent oder variable category gehört (z.B.factor mit 2 levels)) 

#-----------------
# Daten einlesen mit passender Funktion, z.B.  - oder neuen data frame in R bauen, der die gerade beschrieben Spalten enthält
dat.lmm<- read.csv2()

# Kodierung der Spalten checken, z.B.:
str(dat.lmm)


# > die Spalte, wo der ratingtype (shape vs. conceptual,=fixed effect) und die Bild-ID (=random effect) kodiert ist, muss factor sein, falls nicht dann umcodieren, z.B. so
dat.lmm$ratingType <-as.factor(dat.lmm$ratingType)
dat.lmm$Pic <-as.factor(dat.lmm$Pic)


# > die Spalte, wo die gemessenen Werte drinstehen muss numeric sein, falls nicht dann umkodieren, z.B. so:
dat.lmm$PicPair <- as.numeric(dat.lmm$values)

#-----------------
# Signifikanz-Test für Control Analysis  (att vs. pre-att-results) - für jedes Feature separat

#-----------------
# FORM
#-----------------
# Schritt 0) subset für FORM bilden, z.B.
#__________
form<-subset(dat.lmm, feature=="form")

# Schritt 1) linear mixed effects model bauen: LMM
#__________
# brauchen wir um zu testen: 
# Unterscheiden sich pre-att und attentive Form-Ergebnisse signifikant von einander 
# (z.B. sind die Ergebnisse systematisch postiiver wenn das Bild länger gesehen -> dafür: values ~ ratingType), 

# wenn wir berücksichtigen, dass jedes einzelne Bild spezifische Charakteristika hat, die ebenfalls das Ergebnis beeinflusst haben könnten 
# (ein Bild generell positivere Werte und ein anderes Bild generell niedrigere Werte erhielt = intercepts für Pic -> dafür (1|Pic))

# und wir berücksichtigen, dass der Unterschied in in att vs. preatt Werten für jedes Bild unterschiedlich groß sein kann 
# (= random slopes -> dafür die Erweiterung in der Klammer: (1+ratingType|Pic))?


mod1.f<-lmer(values ~ ratingType + 
               (1+ratingType|Pic),
             data=form, 
#          control=lmerControl(optimizer="bobyqa",                #damit ggf. rumspielen, falls das Model nicht konvergiert
#                             optCtrl=list(maxfun=2e5)),          # ~
           REML=FALSE)


# > das ist dann hoffentlich ein model, das konvergiert und kein singular fit hat, ansonsten muss man dafür sorgen, dass das erfüllt ist


# Schritt 2: Model assumptions testen
#__________
# reicht visual inspection denke ich
plot(fitted(mod1.f), residuals(mod1.f)) # muss annähernd blob-like ausssehen, auf jeden Fall nicht wie eine Kurve oder Trichter
qqPlot(resid(mod1.f)) #muss ungefähr innerhalb der Hilfslinien liegen, ansonsten Daten transformieren
hist(residuals(mod1.f)) # muss annähernd normal verteilt sein

# falls nicht erfüllt: Daten transformieren

# ZIEL: das Modell(d.h. eine Datentransformation für values) finden, das möglichst keine Annahme verletzt und mit diesem Modell dann weiter rechnen (transformation könnte man direkt in das Modell reinschreiben z.B. lmer(log(values)~ratingType ...))

#-------------------
# Schritt 3: Signifikanz für ratingType-Effekt berechnen
#__________

# mit anova():
# baue ein Intercept only -Modell, dass sich von dem Modell, das wir auswerten nur dadurch unterscheidet, dass statt ~ratingType jetzt ~1 steht
mod1.f.null<-lmer(values ~ 1 + 
             (1+ratingType|Pic),
           data=form, 
           REML=FALSE)
anova(mod1.f.null, mod1.f)
# > 3 Daten aus dems likelihood ratio test brauchen wir dann (Chisq (df)= , p=), p heißt im Output Pr(>Chisq) oder so ähnlich
# ist vermutl./hoffentlich nicht sign. also p>.05

#-----------------
# ORIENTATION
#-----------------
# Schritt 0) subset für ORIENTATION bilden, z.B.
#__________
orie<-subset(dat.lmm, feature=="orientation")

# Schritt 1) linear mixed effects model bauen: LMM
#__________
# brauchen wir um zu testen: 
# Unterscheiden sich pre-att und attentive Orientation-Ergebnisse signifikant von einander 
# (z.B. sind die Ergebnisse systematisch postiiver wenn das Bild länger gesehen -> dafür: values ~ ratingType), 

# wenn wir berücksichtigen, dass jedes einzelne Bild spezifische Charakteristika hat, die ebenfalls das Ergebnis beeinflusst haben könnten 
# (ein Bild generell positivere Werte und ein anderes Bild generell niedrigere Werte erhielt = intercepts für Pic -> dafür (1|Pic))

# und wir berücksichtigen, dass der Unterschied in in att vs. preatt Werten für jedes Bild unterschiedlich groß sein kann 
# (= random slopes -> dafür die Erweiterung in der Klammer: (1+ratingType|Pic))?


mod1.o<-lmer(values ~ ratingType + 
             (1+ratingType|Pic),
           data=orie, 
           #          control=lmerControl(optimizer="bobyqa",                #damit ggf. rumspielen, falls das Model nicht konvergiert
           #                             optCtrl=list(maxfun=2e5)),          # ~
           REML=FALSE)


# > das ist dann hoffentlich ein model, das konvergiert und kein singular fit hat, ansonsten muss man dafür sorgen, dass das erfüllt ist


# Schritt 2: Model assumptions testen
#__________
# reicht visual inspection denke ich
plot(fitted(mod1.o), residuals(mod1.o)) # muss annähernd blob-like ausssehen, auf jeden Fall nicht wie eine Kurve oder Trichter
qqPlot(resid(mod1.o)) #muss ungefähr innerhalb der Hilfslinien liegen, ansonsten Daten transformieren
hist(residuals(mod1.o)) # muss annähernd normal verteilt sein

# falls nicht erfüllt: Daten transformieren

# ERGEBNIS: das Modell(d.h. eine Datentransformation für values) finden, dass möglichst keine Annahme verletzt und mit diesem Modell dann weiter rechnen (transformation könnte man direkt in das Modell reinschreiben z.B. lmer(log(values)~ratingType ...))

#-------------------
# Schritt 3: Signifikanz für ratingType-Effekt berechnen
#__________

# mit anova():
# baue ein Intercept only -Modell, dass sich von dem Modell, das wir auswerten nur dadurch unterscheidet, dass statt ~ratingType jetzt ~1 steht
mod1.o.null<-lmer(values ~ 1 + 
                  (1+ratingType|Pic),
                data=orie, 
                REML=FALSE)
anova(mod1.o.null, mod1.o)
# > 3 Daten aus dems likelihood ratio test brauchen wir dann (Chisq (df)= , p=), p heißt im Output Pr(>Chisq) oder so ähnlich
# ist vermutl./hoffentlich nicht sign. also p>.05

#-----------------
# LINES
#-----------------
# Schritt 0) subset für LINES bilden, z.B.
#__________
line<-subset(dat.lmm, feature=="lines")

# Schritt 1) linear mixed effects model bauen: LMM
#__________
# brauchen wir um zu testen: 
# Unterscheiden sich pre-att und attentive Lines-Ergebnisse signifikant von einander 
# (z.B. sind die Ergebnisse systematisch postiiver wenn das Bild länger gesehen -> dafür: values ~ ratingType), 

# wenn wir berücksichtigen, dass jedes einzelne Bild spezifische Charakteristika hat, die ebenfalls das Ergebnis beeinflusst haben könnten 
# (ein Bild generell positivere Werte und ein anderes Bild generell niedrigere Werte erhielt = intercepts für Pic -> dafür (1|Pic))

# und wir berücksichtigen, dass der Unterschied in in att vs. preatt Werten für jedes Bild unterschiedlich groß sein kann 
# (= random slopes -> dafür die Erweiterung in der Klammer: (1+ratingType|Pic))?


mod1.l<-lmer(values ~ ratingType + 
               (1+ratingType|Pic),
             data=line, 
             #          control=lmerControl(optimizer="bobyqa",                #damit ggf. rumspielen, falls das Model nicht konvergiert
             #                             optCtrl=list(maxfun=2e5)),          # ~
             REML=FALSE)


# > das ist dann hoffentlich ein model, das konvergiert und kein singular fit hat, ansonsten muss man dafür sorgen, dass das erfüllt ist


# Schritt 2: Model assumptions testen
#__________
# reicht visual inspection denke ich
plot(fitted(mod1.l), residuals(mod1.l)) # muss annähernd blob-like ausssehen, auf jeden Fall nicht wie eine Kurve oder Trichter
qqPlot(resid(mod1.l)) #muss ungefähr innerhalb der Hilfslinien liegen, ansonsten Daten transformieren
hist(residuals(mod1.l)) # muss annähernd normal verteilt sein

# falls nicht erfüllt: Daten transformieren

# ERGEBNIS: das Modell(d.h. eine Datentransformation für values) finden, dass möglichst keine Annahme verletzt und mit diesem Modell dann weiter rechnen (transformation könnte man direkt in das Modell reinschreiben z.B. lmer(log(values)~ratingType ...))

#-------------------
# Schritt 3: Signifikanz für ratingType-Effekt berechnen
#__________

# mit anova():
# baue ein Intercept only -Modell, dass sich von dem Modell, das wir auswerten nur dadurch unterscheidet, dass statt ~ratingType jetzt ~1 steht
mod1.l.null<-lmer(values ~ 1 + 
                    (1+ratingType|Pic),
                  data=line, 
                  REML=FALSE)

anova(mod1.l.null, mod1.l)
# > 3 Daten aus dems likelihood ratio test brauchen wir dann (Chisq (df)= , p=), p heißt im Output Pr(>Chisq) oder so ähnlich
# ist vermutl./hoffentlich nicht sign. also p>.05