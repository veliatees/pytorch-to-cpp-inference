#include <iostream>
#include <fstream>
#include <string>



void Model(int i){
    for(; i < 15; i++){ 
        std::cout << i << std::endl;
    }
}

int main(){
    std::ifstream file("biases/bias1.txt");
    if (!file.is_open()) {
        std::cerr << "Error opening file!" << std::endl;
        return 1;
    }
    std::string line;

    while(std::getline(file, line))
    {
        std::cout << line << std::endl;
    }
    file.close();
    Model(9);
}



int abc() {
    int sayi = 9;
    Model(sayi); 
    return 0;
}