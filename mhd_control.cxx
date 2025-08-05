#include <bout/derivs.hxx>         // To use DDZ()
#include <bout/invert_laplace.hxx> // Laplacian inversion
#include <bout/physicsmodel.hxx>   // Commonly used BOUT++ components
#include <bout/monitor.hxx>
#include <bout/field_factory.hxx>
#include <bout/solver.hxx>
#include <boost/math/special_functions/ellint_1.hpp>
#include <boost/math/special_functions/ellint_2.hpp>
#include <bout/globalfield.hxx>
#include <random>
#include <vector>
#include <fstream>


///////////////////////////////////////////////////////
// global variable to save the data for the python script
std::string output_dir = "data";

///////////////////////////////////////////////////////
// Function coil_greens(Rc, Zc, R, Z) added to the
// input expression parser. This calculates the magnetic flux
// due to a single-turn toroidal coil.
 
// Calculate A_y flux at (X,Z) due to a unit current
double coil_greens(double Rc, double Zc, double X, double Z) {

  // Computes the magnetic vector potential A [V s/m] along the orthogonal direction e_y 
  // in a cartesian cooridnate frame, for a single coil located at (Rc, Zc), with a unit current.

  // Notice how, differently from usual definitions given in spherical of cylindrical geometry,
  // we are instead using cartesian coordinates for coherence with the slab geometry of the domain.  

  double xaxis = 1.0; // Lx = 2, so that the domain's midipoint is at x=1.0
  double dx = X - xaxis; // Distance from the axis of symmetry. 
  double absdx = abs(dx);

  if (absdx < 1e-10) { // Avoid division by zero
    dx = 1e-10; 
    absdx = 1e-10;
  }

  double k2 = 4.* absdx * Rc / (SQ(absdx + Rc) + SQ(Z-Zc));

  if (k2 < 1e-10)  k2 = 1e-10; //avoid division by zero
  if (k2 > 1.0 - 1e-10) k2 = 1.0 - 1e-10;

  double mu0 = 4 * M_PI * 1e-7; // Vacuum permeability [H/m]
  double k = sqrt(k2);


  return 2*mu0/(2 * M_PI * k) * sqrt(Rc/absdx) * ((2. - k2)*boost::math::ellint_1(k) - 2.*boost::math::ellint_2(k)) * dx/absdx;//boost takes k. not k2
}


///////////////////////////////////////////////////////
class CoilGenerator : public FieldGenerator {
public:
  CoilGenerator() = default;
  
  CoilGenerator(BoutReal Rc, BoutReal Zc, FieldGeneratorPtr Z, FieldGeneratorPtr X)
    : Rc(Rc), Zc(Zc), Zgen(Z), Xgen(X) {}
  
  BoutReal generate(BoutReal x, BoutReal y, BoutReal z, BoutReal t) override {
    // Calculate X,Z location of this (x,y,z) point
    BoutReal Z = Zgen->generate(x,y,z,t);
    BoutReal X = Xgen->generate(x,y,z,t);

    return coil_greens(Rc, Zc, X, Z);
  }
  FieldGeneratorPtr clone(const std::list<FieldGeneratorPtr> args) override {
    if (args.size() != 4) {
      throw BoutException("coil_greens expects 4 arguments (Rc, Zc, Z, X)");
    }
    // the following retrieves fro the .inp file the values of Rc, Zc, Z and X
    // and returns a CoilGenerator object with these values. argsit is an iterator
    // over the list of arguments. It is frankly terrible readability but it works. 
    auto argsit = args.begin();
    // Coil positions are constants
    BoutReal Rc_new = (*argsit++)->generate(0,0,0,0);
    BoutReal Zc_new = (*argsit++)->generate(0,0,0,0);
    // Evaluation location can be a function of x,y,z,t
    FieldGeneratorPtr Xgen_new = *argsit;
    FieldGeneratorPtr Zgen_new = *argsit++;
    return std::make_shared<CoilGenerator>(Rc_new, Zc_new, Xgen_new, Zgen_new);
  }
private:
  BoutReal Rc, Zc;  // Coil location, fixed
  FieldGeneratorPtr Zgen, Xgen; // Location, function of x,y,z,t
};



class KH : public PhysicsModel {

public:
  Field3D getA(){
    return Apar;
  }


private:
  // Evolving variables
  Field3D omega; ///< Density and vorticity
  Field3D phi;
  Field3D J;
  Field3D Apar;
  Field3D psi_ext;
  Field3D psi;
  Field3D Eext; // electric field from changing magnetic flux

  // generators
  FieldGeneratorPtr psiext_gen;

  // Constants
  BoutReal viscosity;
  BoutReal resistivity;
  BoutReal density;
  BoutReal mu0;

  // control variables
  BoutReal intensity = 0.0;
  BoutReal next_intensity;
  BoutReal m = 0.0;
  BoutReal q = 0.0;

  std::unique_ptr<Laplacian> phiSolver; ///< Performs Laplacian inversions to calculate phi
  std::unique_ptr<Laplacian> AparSolver;

protected:
  int init(bool UNUSED(restarting)) {
    // ****************** Adding the coil generator *************
    // Add a function which can be used in input expressions
    // This calculates the Greens function for a coil
    FieldFactory::get()->addGenerator("coil_greens", std::make_shared<CoilGenerator>());

    /******************Reading options *****************/
    // Get the options
    auto& options = Options::root();
    viscosity = options["mhd"]["viscosity"].doc("viscosity").withDefault(0.001);
    resistivity = options["mhd"]["resistivity"].doc("resistivity").withDefault(0.001);
    density = options["mhd"]["density"].doc("density").withDefault(1.0);
    mu0 = options["mhd"]["mu0"].doc("Vacuum permeability [H/m]").withDefault(4 * M_PI * 1e-7);

    // External flux Psiext and toroidal electric field Eext can vary in time
    // so a generator is needed which can be evaluated at each step
    FieldFactory* factory = FieldFactory::get();
    psiext_gen = factory->parse(
        options["mhd"]["psiext"].doc("External magnetic potential [V s/m]").withDefault("0.0"), &options);
    psi_ext = factory->create3D(psiext_gen);

    /************ Create a solver for potential ********/
     phiSolver = Laplacian::create(&options["phiSolver"]);
     AparSolver = Laplacian::create(&options["AparSolver"]);

    /************ Tell BOUT++ what to solve ************/
    SOLVE_FOR(omega, Apar);

    phi = phiSolver->solve(omega);
  
    // Output phi, omega, psi, J
    SAVE_REPEAT(phi, omega, Apar, J, intensity);

    return 0;
  }

  int rhs(BoutReal time) {

    // Run communications
    ////////////////////////////////////////////////////////////////////////////
    mesh->communicate(Apar, omega);


    // solve for phi
    phi = phiSolver->solve(omega);
    // solve for Apar
    J = -Delp2(Apar);

    ////////////////////////////////////////////////////////////////////////////
    mesh->communicate(phi, J);

    ////////////////////////////////////////////////////////////////////////////
    FieldFactory* factory = FieldFactory::get();
    psi_ext = factory->create3D(psiext_gen, bout::globals::mesh, CELL_CENTRE, time);
    
    mesh->communicate(psi_ext);

    // Poloidal flux, including external field
    // piecewise linear with control fed every 0.1 seconds
    psi = Apar + (m*time + q)*psi_ext; //this implies the presence of an electric field varying in time 

    // to avoid that let us assume that the coils change instantanously so no electric field is forced
    // BoutReal coeff = time*time*a + b*time + c;
    //psi = Apar + next_intensity*psi_ext; // this is the poloidal flux, psi = Aphi + psi_ext
    // psi = Apar + coeff * psi_ext; // this is the poloidal flux, psi = Aphi + psi_ext
    // notice that for controls that are piecewise linear in time we would have an additional external electric field
    // that is induced by the time varying external magnetic field. 
    // It would look like: Eext = -dt(A) 

    Eext = -m*psi_ext;

    mesh->communicate(psi, Eext);

    // Vorticity evolution
    ddt(omega) = - bracket(phi, omega, BRACKET_ARAKAWA) // advection
                 + viscosity * Delp2(omega)           // Viscous diffusion term]
                 - bracket(psi, J, BRACKET_ARAKAWA) // the magnetic field B is rescaled by sqrt mu0. 
                 // thuse we are actually using a rescaled force.
                 ; 
        

    // Apar evolution:
    ddt(Apar) = bracket(psi, phi, BRACKET_ARAKAWA)
                - resistivity * J //ohmic dissipation
                + Eext //or maybe DDT(psi_ext)
                ;           

    return 0;
  }
 
  int outputMonitor(BoutReal simtime, int iter, int NOUT) {
    if (iter % 10 != 0) return 0;

    int rank = 0;
    #ifdef BOUT_HAS_MPI
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      MPI_Barrier(MPI_COMM_WORLD);
    #endif

    // dataSaver() ust not be inside the rnak==0, otherwise cannot aggregate data
    // as it is not called by all ranks.
    dataSaver();
  
    if (rank == 0) {
      
        output.write("\nOutput monitor, time = {:e}, step {:d} of {:d} INPUT\n", simtime, iter,
                  NOUT);

        std::cout.flush();

        m = (intensity-next_intensity)/0.1; // 0.1 is the time step for the control input
        q = intensity - m*simtime;
  
        intensity = next_intensity;

        dataReader();
    }

    #ifdef BOUT_HAS_MPI
    // Broadcast to all ranks 
      MPI_Bcast(&next_intensity, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MPI_Bcast(&m, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MPI_Bcast(&q, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    #endif
    return 0;
 }

  int dataSaver() {
    int rank = 0;
    #ifdef BOUT_HAS_MPI
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    #endif

    mesh->communicate(omega, phi, Apar, psi_ext, J);

    GlobalField3D g3d(mesh);
    g3d.gather(J);

    GlobalField3D vyfield(mesh);
    vyfield.gather(DDZ(phi));

    if (rank == 0) {
        auto &arr = g3d.getData();
        std::ofstream outfile(output_dir + "/data.bin", std::ios::binary);
        outfile.write(reinterpret_cast<const char*>(&arr[0]), arr.size() * sizeof(double));
        outfile.close();

        // now saving the x component of phi
        auto &arr_vy = vyfield.getData();
        std::ofstream outfile_vy(output_dir + "/vx.bin", std::ios::binary);
        outfile_vy.write(reinterpret_cast<const char*>(&arr_vy[0]), arr_vy.size() * sizeof(double));
        outfile_vy.close();
    }

    return 0;
}

  void dataReader(){
    int rank = 0;
    #ifdef BOUT_HAS_MPI
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    #endif
    if (rank == 0) {
      std::cout.flush();
      std::cerr.flush();
      if (!(std::cin >> next_intensity)) std::cerr << "No input received. Exiting."<<std::endl<<std::flush;
      else output.write("New intensity: {}\n", next_intensity);
    }
  }
};


int main(int argc, char **argv) {
  // Parse -d argument
  for (int i = 1; i < argc - 1; ++i) {
    if (std::string(argv[i]) == "-d") {
      output_dir = argv[i + 1];
    }
  }
  BoutInitialise(argc, argv);

  KH *model = new KH(); 

  auto solver = Solver::create(); 
  solver->setModel(model); 
  solver->solve(); // Run the solver

  delete model;
  BoutFinalise(); 
  return 0;
}