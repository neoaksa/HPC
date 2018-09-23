#include <iostream>
#include <fstream>
#include <omp.h>
#include <vector>
using namespace std;



#define x 300
#define y 300
#define z 300
// define 3D vector
typedef vector<int> Dim1;
typedef vector<Dim1> Dim2;
typedef vector<Dim2> Dim3;
int* createParticle(Dim3);
bool moveParticles(Dim3&, int*);

int main(int argc, char *argv[])
{
	// define a lattice and set value to zero
	Dim3 lattice(x,Dim2(y,Dim1(z,0)));
	vector<string> points((int)(x*y*z/9)); // save points
	// total sticked point
	int total = 1;
	// set orignal point in the central
	lattice[(int)(x/2)][(int)(y/2)][(int)(z/2)] = 1;
	// set max threads
	omp_set_num_threads (8);
	
	#pragma omp parallel
	{
		while(total < (int)(x*y*z/10)){
			//create a particle
			int *point = createParticle(lattice);
			//move particle
			if(moveParticles(lattice, point)){
				#pragma omp critical
				{
					total++;
					//~ cout << point[0] << "|" << point[1] << "|" << point[2] << endl;
					lattice[point[0]][point[1]][point[2]] = 1;
					points.push_back(std::to_string(point[0])+","+std::to_string(point[1])+","+std::to_string(point[2])+",\n");
					//~ cout << total << endl;
				}
			}
		}
	}
	// export array
	ofstream myfile;
	myfile.open ("crystal.csv");
	for(std::vector<string>::iterator it = points.begin(); it != points.end(); ++it) {
		myfile << *it;
	}
	myfile.close();
	return 0;
}

// create a particle
int* createParticle(Dim3 lattice)
{
	// flag for exiting point 
	bool flag = true;

	while(flag){
		//create a random position
		int new_x = rand()%(x);
		int new_y = rand()%(y);
		int new_z = rand()%(z);

		//check if exit
		if(lattice[new_x][new_y][new_z] == 0 ){
			flag = false;
			int* point = new int[3]{new_x,new_y,new_z};
			return point;
		}
	}
	return NULL;
}

// move particles
bool moveParticles(Dim3& lattice, int* point)
{
	// random move around point
	while(1){
		// check around point
		for(int i = -1; i < 2; i++){
			for(int j = -1; j < 2; j++){
				for(int k = -1; k < 2; k++){
					if((point[0]+i<x && point[0]+i>=0) &&
						(point[1]+j<y && point[1]+j>=0) &&
						(point[2]+k<z && point[2]+k>=0)){
						if(lattice[point[0]+i][point[1]+j][point[2]+k]==1){
							//~ lattice[point[0]][point[1]][point[2]] = 1;
							return true;
						}
					}
				}
			}
		} 
		// randomly move
		int move_x = rand()%(3) - 1;
		int move_y = rand()%(3) - 1;
		int move_z = rand()%(3) - 1;
		//move out 
		if(point[0]+move_x == x  || point[0]+move_x < 0 ||
			point[1]+move_y == y || point[1]+move_y < 0 ||
			point[2]+move_z == z || point[2]+move_z < 0){
			return false;
		}
		// move next
		else {
			point[0] = point[0]+move_x;
			point[1] = point[1]+move_y;
			point[2] = point[2]+move_z;
		}
	}
}
