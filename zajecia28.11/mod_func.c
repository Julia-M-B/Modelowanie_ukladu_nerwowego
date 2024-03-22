#include <stdio.h>
#include "hocdec.h"
#define IMPORT extern __declspec(dllimport)
IMPORT int nrnmpi_myid, nrn_nobanner_;

extern void _CaT_reg();
extern void _htc_reg();

void modl_reg(){
	//nrn_mswindll_stdio(stdin, stdout, stderr);
    if (!nrn_nobanner_) if (nrnmpi_myid < 1) {
	fprintf(stderr, "Additional mechanisms from files\n");

fprintf(stderr," CaT.mod");
fprintf(stderr," htc.mod");
fprintf(stderr, "\n");
    }
_CaT_reg();
_htc_reg();
}
