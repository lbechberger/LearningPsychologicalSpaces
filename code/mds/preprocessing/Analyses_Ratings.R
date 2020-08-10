# Das ist ein Orientierungs-Skript für die Auswertung der shape und conceptual similarity ratings (control analysis 1+2)
# Namen von Spalten & Co müssen an den konkreten Datensatz angepasst werden.
#-----------------
# CONTENT of the Skript
# für control analysis 1: shape und concept. ratings sign unterschiedlich?        - ab line 43
# für control analysis 1: nur between-Pairs: shape und concept unterschiedl?      - ab line 111
# für control analysis 1: nur within-pairs: shape und concept unterschiedl?       - ab line 156
# für control analysis 2: mean vs. median shape-results sign unterschl? (t-Test)  - ab line 232
# for control analysis 1: Spearman correlation zu shape vs concept  ratings       - ab line 270

#-----------------
# PACKAGES (ggf. install.packages() vorher)
library(ordinal)        # für alle clmm-models
library(RVAideMemoire)  # für die Anova()
library(car)            # für die Anova()
library(pastecs)        # für normalverteilungstest (t-Test)
library(ggplot2)        # für Plots - falls man die in R erstellen will, damit das einheitlich ist
#-----------------
# Mixed-Models: benötigtes csv-FILE (enhält am besten keine NAs, ansonsten muss das in den späteren Funktionen berücksichtigt werden):
# Data file sollte die folgenden Spalten mind. haben (für jeden gemessenen Rating-Wert eine Zeile!):
# PicPair:        eine Spalte, wo kodiert ist, um welches Bildpaar es geht (factor mit 1770 levels)
# PicPair_Type :  eine Spalte, wo kodiert ist, ob es ein within- oder between-category pair ist (z.B. factor mit 2 levels, oder anderer Type)
# visualType:     eine Spalte, wo kodiert ist, ob beide Objekte eines Pairs coherent oder variable oder mixed(bei between pairs) sind (z.B. factor mit 3 levels, oder anderer Type)
# ratingType:     eine Spalte, wo kodiert ist, ob es sich um ein shape oder ein conceptual rating handelt (factor mit 2 levels)
# ratings:        eine Spalte, wo die ratings selbst drin stehen (ordered factor; 1 muss bei beiden ratingsTypes für das gleiche stehen, z.B. 1=dissimilar, 5=similar)

#-----------------
# Daten einlesen mit passender Funktion, z.B.  
data<- read.csv2()

# Kodierung der Spalten checken, z.B. mit
str(data)

# > die Spalte wo die abgegebenen Ratings drinstehen, muss Ord.factor sein mit 5 levels, falls nicht dann umcodieren, z.B. so:
data$ratings <- factor(data$rating, ordered=T)

# > die Spalte, wo der ratingtype (shape vs. conceptual) kodiert ist (=fixed effect) muss factor sein, falls nicht dann umcodieren, z.B. so
data$ratingType <-as.factor(data$ratingType)

# > die Spalte, wo die item pairs kodiert sind (= random effect) muss factor sein, falls nicht dann umkodieren, z.B. so:
data$PicPair <- as.factor(data$PicPair)

#-----------------
# Signifikanz-Test für Control Analysis 1 (shape vs. concept. ratings)

#-----------------
# Schritt 1) mixed effects model für ordinal data bauen: CLMM
#__________
# brauchen wir um zu testen: 
# Unterscheiden sich shape und conceptual ratings signifikant von einander 
# (z.B. sind die ratings bei einem RatingType systematisch niedriger als bei dem anderen -> dafür: ratings ~ ratingType), 

# wenn wir berücksichtigen, dass jedes einzelne Bildpaar spezifische Charakteristika hat, die ebenfalls das Rating beeinflusst haben könnten 
# (ein Bildpaar generell höhere ratings und ein anderes Bildpaar generell niedrigere Ratings erhielt = intercepts für PicPair -> dafür (1|PicPair))

# und wir berücksichtigen, dass der Unterschied in shape und conceptual ratings für jedes Bildpaar unterschiedlich groß sein kann 
# (= random slopes -> dafür die Erweiterung in der Klammer: (1+ratingType|PicPair))?


model1<-clmm(ratings ~ ratingType + 
               (1+ratingType|PicPair),
             data=data)


# > das ist dann hoffentlich ein model, das konvergiert und kein singular fit hat, ansonsten muss man dafür sorgen, dass das erfüllt ist
# (man könnte noch einiges in das Model packen, habe ich aber zu wenig Ahnung von ...)
#---------
# ALTERNATIV: Model ohne random slopes (ich habe gelesen, dass random slopes mittlerweile geht und auch das sie nicht verfügbar sind, also mal testen)
# für model ohne random slopes: clmm() oder clmm2(), random factor könnten eventl. auch als random=PicPair zu schreiben sein
model2<-clmm(ratings ~ ratingType + 
               (1|PicPair),
             data=data)

# -----------------
# Schritt 2: Model assumptions testen
#__________
# (hab ich wenig Ahnung, aber die folgenden 2 Sache gefunden als angeblich  wichtig, weil sonst Ergebnisse aus dem Modell nicht verlässlich)
# es kann sein dass beide Tests nicht für clmm mit random slopes (model1) anwendbar sind, dann mit model2 testen

# 1) (partial) proportional odds/equal slopes (= der Effekt von ratingType muss konstant sein für jeden Anstieg im Rating-Wert 1vs.2, 2vs.3, etc)
nominal_test(model1)
#> annahme ist erfüllt, wenn p>.05 (als kein sign. Ergebnis)

# 2) keine scale effects (= Skalen-Punkte wurden in beiden Studien und für alle Bildpaare gleich interpretiert und auch genutzt, also nicht in manchen Fällen z.B. Extrempunkte gemieden, nur mittelster Wert gewählt etc.)
scale_test(model1)
# > Annahme ist erfüllt, wenn p>.05 (also kein sign. Ergebnis)
# falls nicht erfüllt: ratings-Spalte transformieren z.B. z-Transormation versuchen, 
# die transformierten Ratings dann als neue Spalte in den Datensatz einfügen, 
# diese Spalte dann wieder als ord.Fact kodieren, 
# das clmm-Model mit TRANSFratings ~ ratingType ...) rechnen und für dieses neue Modell dann den scale_test erneut rechnen

# ZIEL: das Modell finden, dass möglichst nichts verletzt und mit diesem Modell dann weiter rechnen (allerdings in jedem Fall dann die random slopes(falls verfügbar) wieder reinpacken)

#-------------------
# Schritt 3: Signifikanz für ratingType-Effekt berechnen
#__________
# habe ich beides gefunden mit anova(), bei Christensen selbst der das ordinal-Package gemacht hat (ist auch das was ich bisher kannte), und Anova() (analysis of deviance, bei Bross in einem Tutorial zu rating-Auswertungen)

# für die anova() [würde ich glaube ich machen]:
# baue ein Intercept only -Modell, dass sich von dem Modell, das wir auswerten nur dadurch unterscheidet, dass statt ~ratingType jetzt ~1 steht
model1.null <- model1<-clmm(ratings ~ 1 +
                             (1+ratingType|PicPair),
                           data=data)
anova(model1.null, model1)
# > 3 Daten aus dems likelihood ratio test brauchen wir dann (Chisq (df)= , p=), p heißt im Output Pr(>Chisq) oder so ähnlich; (die Levels of interest sind: p<.05, <.01. <.001 und <.0001)

# für die Anova()
Anova(model1, type="II")
# > die gleichen 3 Daten  (Chisq (df)= , p=) brauchen wir dann; (die Levels of interest sind: p<.05, <.01. <.001 und <.0001)

#-----------------
# Signifikanz-Test für Control Analysis 1 (shape vs. concept. ratings in BETWEEN-category pairs)

#-----------------
# Schritt 1) subset aus dem Data frame bilden, wo nur noch between PicPairs drin sind, z.B. so (Spaltennamen und level-Names ggf. anpassen!)
#__________
between <- subset(data, PicPair_Type=="between")

#-----------------
# AB HIER GLEICHE PROZEDURE WIE OBEN für den gesamt Datensatz, nur jetzt den kleineren between-Datensatz analysieren
# Schritt 2) clmm-Model bauen
#__________

model1.betweenn <-clmm(ratings ~ ratingType + 
                         (1+ratingType|PicPair),
                       data=between)

#---------
# ALTERNATIV: Model ohne random slopes 
model2.between <- clmm(ratings ~ ratingType + 
                         (1|PicPair),
                       data=between)

# -----------------
# Schritt 3: Model assumptions testen
#__________
# s. oben, darf nichts signifikant werden, sonst z.B.Daten transformieren 
nominal_test(model1.between)
scale_test(model1.between)

# ZIEL: das Modell finden, dass möglichst nichts verletzt und mit diesem Modell dann weiter rechnen (allerdings in jedem Fall dann die random slopes wieder reinpacken)

#-------------------
# Schritt 4: Signifikanz für ratingType-Effekt berechnen
#__________
# s. oben, z.B.  anova():
# baue ein Intercept only -Modell, dass sich von dem Modell, das wir auswerten nur dadurch unterscheidet, dass statt ~ratingType jetzt ~1 steht

model1.between.null <-clmm(ratings ~ 1 + 
                         (1+ratingType|PicPair),
                       data=between)
anova(model1.between.null, model1.between)
# > 3 Daten aus dems likelihood ratio test brauchen wir dann (Chisq (df)= , p=); (die Levels of interest sind: p<.05, <.01. <.001 und <.0001)

#-----------------
# Signifikanz-Test für Control Analysis 1 (shape vs. concept. ratings in WITHIN-category pairs)

#-----------------
# Schritt 1) subset aus dem Data frame bilden, wo nur noch within PicPairs drin sind, z.B. so (Spaltennamen und level-Names ggf. anpassen!)
#__________
within <- subset(data, PicPair_Type=="within")

#-----------------
# AB HIER ÄHNLICHE PROZEDURE WIE OBEN für den gesamt Datensatz
# Schritt 2) clmm-Model bauen, jetzt mit 2 fixed effects: ratingType und visualType (* erlaub auch Interaktion; falls nicht zulässig, dann mit mit Haupteffekten (+) rechnen)
#__________

model1.within <-clmm(ratings ~ ratingType * visualType +
                         (1+ratingType|PicPair),
                       data=within)

#---------
# ALTERNATIV: Model ohne random slopes 
model2.within <- clmm(ratings ~ ratingType * visualType +
                        (1|PicPair),
                      data=within)

# -----------------
# Schritt 3: Model assumptions testen
#__________
# s. oben, darf nichts signifikant werden, sonst z.B.Daten transformieren 
nominal_test(model1.within)
scale_test(model1.within)


# ZIEL: das Modell finden, dass möglichst nichts verletzt und mit diesem Modell dann weiter rechnen (allerdings in jedem Fall dann die random slopes wieder reinpacken)

#-------------------
# Schritt 4: Signifikanz für ratingType-Effekt und visualType-Effekt berechnen
#__________
# s. oben, z.B. mit anova():

# A) Interaktion sign?
# baue ein Null-Modell für die Interaktion (statt *, + schreiben, basieren auf dem Modell, das die meisten Anforderungen erfüllt und wieder random slopes einschließt)

model1.within.ohneInterakt <-clmm(ratings ~  ratingType + visualType +
                                   (1+ratingType|PicPair),
                                 data=within)
anova(model1.within.ohneInterakt, model1.within)
# > 3 Daten aus dems likelihood ratio test brauchen wir dann (Chisq (df)= , p=); (die Levels of interest sind: p<.05, <.01. <.001 und <.0001)


# B) Haupteffekt für ratingType sign?
# baue ein Null-Modell für den Faktor ratingType (ratingType von der Fixed effects structure entfernen von dem Modell (hier model1.within), dass die meisten Anforderungen erfüllt und wieder random slopes einschließt)

model1.within.ohneRatingT <-clmm(ratings ~  visualType +
                                   (1+ratingType|PicPair),
                                 data=within)
anova(model1.within.ohneRatingT, model1.within)
# > 3 Daten aus dems likelihood ratio test brauchen wir dann (Chisq (df)= , p=); (die Levels of interest sind: p<.05, <.01. <.001 und <.0001)

# C) Haupteffekt für visualType sign?
# baue ein Null-Modell für den Faktor visual Type (jetzt visualType von der Fixed effects structure entfernen von dem Modell (hier model1.within), dass die meisten Anforderungen erfüllt und wieder random slopes einschließt)

model1.within.ohneVisualT <-clmm(ratings ~  ratingType +
                                   (1+ratingType|PicPair),
                                 data=within)
anova(model1.within.ohneVisualT, model1.within)
# > 3 Daten aus dems likelihood ratio test brauchen wir dann (Chisq (df)= , p=); (die Levels of interest sind: p<.05, <.01. <.001 und <.0001)

# ACHTUNG!
# wenn die Interaktion nicht signifikant war, dann haben wir ein anderes Basis-Modell, nämlich: model1.within.ohneInterakt
# die Analysen zu den Haupteffekten würden dann model1.within.ohneInterakt als Vergleichsmodell nehmen
# d.h. wir rechnen:
anova(model1.within.ohneRatingT, model1.within.ohneInterakt)
anova(model1.within.ohneVisualT, model1.within.ohneInterakt)

# >> ggf. muss für die within-Daten dann noch mehr gerechnet werden, kommt auf die Ergebnisse an

#-----------------
#-----------------
# mean vs median: benötigtes csv-FILE (enhält am besten keine NAs, ansonsten muss das in den späteren Funktionen berücksichtigt werden):
# Data file (only shape _SIMILARITY_ ratings) sollte die folgenden Spalten mind. haben (für jeden _aggregierten_ Wert eine Zeile! - das ist anders als bei den Mixed Models):
# PicPair:          eine Spalte, wo kodiert ist, um welches Bildpaar es geht (factor mit 1770 levels)
# aggregationType:  eine Spalte, wo kodiert ist, wie die verschiedenen Probanden-Werte zusammengefasst wurden, per mean oder median (factor mit 2 levels)
# values:           eine Spalte, wo die aggregierten similarity-Werte drin stehen (numeric variable; 1 muss bei beiden ratingsTypes für das gleiche stehen, z.B. 1=dissimilar, 5=similar)

#-----------------
# Daten einlesen mit passender Funktion, z.B. wäre möglich - oder neuen data frame in R so bauen, wie gerade beschrieben
data.agg<- read.csv2()

# Kodierung der Spalten checken, z.B.:
str(data.agg)

# > die Spalte wo der Aggregierungs-Typ drinstehen, muss factor sein (=fixed effect) , falls nicht dann umcodieren, z.B. so
data.agg$aggregationType <-as.factor(data.agg$aggregationType)

# > die Spalte, wo die aggregierten Werte drin stehen muss numerisch sein, falls nicht dann umkodieren, z.B. so:
data.agg$values <- as.numeric(data.agg$values)

#-----------------
# Signifikanz-Test für Control Analysis 2 (mean vs. median shape werte)

#-----------------
# für Mixed Models sind zuwenig Daten (haben nur exakt 1 Wert je case, wenn ich da kein Denkfehler habe), deshalb (item-basierter) t-Test
# rechnen dependend t-Test (weil es für jedes PicPair einen Mean und einen Median gibt)

# assumptions testen: sowohl für die means als auch die mediane muss normalverteilung vorliegen, visual inspection reicht, oder so:
by(data.agg$values, data.agg$aggregationType, stat.desc, basic=FALSE, norm=TRUE) 
#bei beiden aggregationType levels sollte der p-Wert zum Norm-Test (ganz am Ende der Tabelle) >.05 sein, also nicht sign, dann ist Annahme erfüllt
# fall Annahme verletzt, z.B. Daten transformieren

# Signifikanz testen:
t.test(values ~ aggregationType, data=data.agg, paired=T)

# > 3 Daten aus dem test brauchen wir  (t (df)= , p=); (die Levels of interest sind: p<.05, <.01. <.001 und <.0001)


#-----------------
# Korrelationsanalysen: benötigtes csv-FILE (enthält am besten keine NAs, ansonsten muss das in den späteren Funktionen berücksichtigt werden):
# Data file sollte die folgenden Spalten mind. haben (für jedes Bild-Paar eine Zeile! - das ist anders als bei den MixedModels und dem t-Test!):
# PicPair:        eine Spalte, wo kodiert ist, um welches Bild-Paar es geht (factor mit 1770 levels)
# shapeRatings:   eine Spalte, wo der BildPaar-median aus den shape data drinsteht (numeric variable)
# conceptRatings: eine Spalte, wo der BildPaar-median aus den conceptual data drinsteht (numeric variable)

#-----------------
# Daten einlesen mit passender Funktion, z.B. wäre möglich - oder neuen data frame in R bauen, mit den gerade genannten Spalten
data.corr<- read.csv2()

# Kodierung der Spalten checken, z.B.:
str(data.corr)

# > all 2 Spalten wo die BildPaar-mediane drin stehen müssen numeric sein, falls nicht dann umcodieren, z.B. so:
data.corr$shapeRatings <- as.numeric(data.corr$shapeRatings)
#-----------------
# Spearman-Korelation für control analysis 1:  Korrelation zwischen shape und concept. ratings? 

#-----------------
# cor.test() gehört zum base system von R
# kann für Pearson + Spearman-Correlations genutzt werden
# gibt: r-Werte, p-Werte
# wir rechnen two-tailed (weil wir keine spezielle Hypothese haben)
#-----------------

#__________Assumptions checken
# alle samples (alle daten aus den 2 numeric ratingType Spalten) müssen jeweils annähernd normal verteilt sein (reicht visual inspection, denke ich)

#__________Korrelation berechnen

cor.test(data.corr$shapeRatings, data.corr$conceptRatings, method="spearman")

# > folgende Daten brauchen wir dann:
# r (roh aus sample estimates), auf 2 Nachkommastellen gerundet
#p< .. (die Levels of interest sind: <.05, <.01. <.001 und <.0001)

