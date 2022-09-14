library(nVennR)

args <- commandArgs(trailingOnly = TRUE)

csv_file <- args[1]
out_file <- args[2]

candid <- read.csv(csv_file, sep=";")

CT <- subset(candid, Chest.Tube == 1)$img_id
PTX <- subset(candid, Pneumothorax == 1)$img_id
RF <- subset(candid, Rib.Fracture == 1)$img_id

num_entries_ctp <- length(candid$Chest.Tube.Prediction)

if (num_entries_ctp > 0) {
  print("in if ...")
  CTP <- subset(candid, Chest.Tube.Prediction == 1)$img_id
  myV <-plotVenn(list("Chest Tube"=CT, "Pneumothorax"=PTX, "Rib Fracture"=RF, "Chest Tube Prediction"=CTP), borderWidth = 3, setColors=c('#9ED9A4', '#9BC1E2', '#F9F18B', '#D84315'), labelRegions = F, fontScale = 1.5, nCycles = 2000, outFile=out_file)
} else {
  print("in else ...")
  CT
  myV <-plotVenn(list("Chest Tube"=CT, "Pneumothorax"=PTX, "Rib Fracture"=RF), borderWidth = 3, setColors=c('#9ED9A4', '#9BC1E2', '#F9F18B'), labelRegions = F, fontScale = 1.5, nCycles = 2000, outFile=out_file)
}
  # '#9ED9A4'
# showSVG(nVennObj = myV, opacity = 0.1, borderWidth = 3)
# showSVG(nVennObj = myV, setColors = c('#d7100b', 'teal', 'yellow', 'black', '#2b55b7'))
# showSVG(nVennObj = myV, opacity = 0.1, labelRegions = F, fontScale = 3)
# myV4 <- plotVenn(list(CT=CT, PTX=PTX, RF=RF), nCycles = 2000, setColors=c('red', 'green', 'blue'), labelRegions=F, fontScale=2, opacity=0.2, borderWidth=2)

