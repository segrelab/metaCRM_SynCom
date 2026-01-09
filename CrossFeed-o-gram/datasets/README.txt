Details about interaction terms are desribed below.

interaction-terms_experimental.csv: 
    Terms derived from the experimental exometabolomics data. 
    
interaction-terms_CRMgznorm.csv:
    Terms that are derived from the CRM parameters (more description in relevant notebooks & manuscript). This file is z-score normalized to bring
        positive interaction terms (production) and negative interaction terms (consumption) closer to each other, as the values are very
        different (by default). These were calculated from the interactions that were normed by growth, then combined into one file by adding by the
        smallest value (positive) or subtracting by the largest value (negative) and saved to one csv.

interaction-terms_CRMgnorm.csv
    Terms that are derived from the CRM parameters (more description in relevant notebooks & manuscript). This file is normalized by the growth terms,
        as species with low representation in the final communities had outsized contributions of metabolites.
        
interaction-terms_CRM.csv
    Terms derived from the CRM parameters (more description in relevant notebooks & manuscript). 
