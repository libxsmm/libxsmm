
#include "constants.hpp"

namespace edge {
  namespace io {
    class Receivers;
  }
}

class edge::io::Receivers {
  public:
    double getRecvTimeRel(unsigned int i_var1, double i_var2, double i_var3) { return 0.0; }
    void writeRecvAll(unsigned int i_var1, real_base i_var2[N_QUANTITIES][N_ELEMENT_MODES][N_CRUNS]) { return; }
};
