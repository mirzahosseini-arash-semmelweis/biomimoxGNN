library(dplyr)

# create Percepta descriptor table
PerceptaSMILES <- read.delim(
  file.path("percepta_files", "PerceptaSMILES.txt"),
  stringsAsFactors = FALSE
)
PerceptaRefractivity <- read.delim(
  file.path("percepta_files", "PerceptaRefractivity.txt"),
  stringsAsFactors = FALSE
)
PerceptalogD <- read.delim(
  file.path("percepta_files", "PerceptalogD.txt"),
  stringsAsFactors = FALSE
)
PerceptaNeg <- read.delim(
  file.path("percepta_files", "PerceptaNeg.txt"),
  stringsAsFactors = FALSE
)
PerceptaPos <- read.delim(
  file.path("percepta_files", "PerceptaPos.txt"),
  stringsAsFactors = FALSE
)
smiles_gnn <- read.csv(
  file.path("input_files", "SMILESforGNN.csv"),
  stringsAsFactors = FALSE
)

# check row numbers
stopifnot(
  nrow(PerceptaSMILES) == nrow(smiles_gnn),
  nrow(PerceptaSMILES) == nrow(PerceptaRefractivity),
  nrow(PerceptaSMILES) == nrow(PerceptalogD),
  nrow(PerceptaSMILES) == nrow(PerceptaNeg),
  nrow(PerceptaSMILES) == nrow(PerceptaPos)
)

# check that SMILES are already in same order
stopifnot(all(PerceptaSMILES$SMILES == smiles_gnn$SMILES))

# assemble final descriptor table
percepta_tbl <- tibble(
  refractivity  = PerceptaRefractivity$Molar.Refractivity,
  logD          = PerceptalogD$LogD..pH...7.40.,
  neg           = PerceptaNeg$Neg,
  pos           = PerceptaPos$Pos
)

write.csv(
  percepta_tbl,
  file.path("input_files", "percepta_desc.csv"),
  row.names = FALSE
)
