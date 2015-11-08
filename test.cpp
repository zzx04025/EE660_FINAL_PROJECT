#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <math.h>   
#include <fstream> 
#include <vector> 
#include <string>

using namespace std;

int main(){
int trainN = 1017209;
ifstream ex1;
ex1.open("/Users/Kathy/workspace/EE660_FINAL_PROJECT/store_lwy.csv");
vector<vector<int> > table_store; 

// read store.cvs into table_store
for(int i = 1; i <= 1115; ++i){
	vector<int> v; 
	string cell;
	char Filename[1000];
	ex1 >> cell; // read a line from csv
//cout << cell<<endl;

	for (int j=0;j<=cell.size();j++)
        {
            Filename[j]=cell[j]; 
        }
  char * pch;
 	pch = strtok (Filename,",."); 
 	while (pch != NULL)
  {
  //printf ("%s\n",pch);
     if(string(pch).size() >= 6){
	int month1,month2,month3,month4;
	sscanf(pch,"%d/%d/%d/%d",&month1,&month2,&month3,&month4);
	v.push_back(month1);
	v.push_back(month2);
	v.push_back(month3);
	v.push_back(month4);

}

  else	v.push_back(atoi(pch));
    pch = strtok (NULL, ",.");

  }

  table_store.push_back(v);
  delete[] pch;


}
  

//   for(int i = 0; i < table_store.size(); ++i){
// 	for(int j = 0; j < table_store[i].size(); ++j){
// 		cout << table_store[i][j] << " ";
// 	}
// 	cout<<endl;
// }



ifstream ex2;
ex2.open("/Users/Kathy/workspace/EE660_FINAL_PROJECT/train_lwy.csv");
vector<vector<int> > table_train; 

// read store.cvs into table_store
for(int i = 1; i <= trainN; ++i){
	vector<int> v; 
	string cell;
	char Filename[1000];
	ex2 >> cell; // read a line from csv
//cout << cell<<endl;

	for (int j=0;j<=cell.size();j++)
        {
            Filename[j]=cell[j]; 
        }
  char * pch;
 	pch = strtok (Filename,",./"); 
 	while (pch != NULL)
  {
// printf ("%s\n",pch);
//cout  << atoi(pch) << endl;
if(string(pch).size() >= 6){
	int day,month,year;
	sscanf(pch,"%d/%d/%d",&day,&month,&year);
	v.push_back(day);
	v.push_back(month);
	v.push_back(year+2000);
}
 // else   
  else	v.push_back(atoi(pch));
    pch = strtok (NULL, ",.");

  }

  table_train.push_back(v);
  delete[] pch;

}
 
// for(int i = 0; i < table_train.size(); ++i){
// 	for(int j = 0; j < table_train[i].size(); ++j){
// 		cout << table_train[i][j] << " ";
// 	}
// 	cout<<endl;
// }




////////////////// compare
vector<int> inComp(1115,0);// competetion
vector<int> inPromo(1115,0);// competetion
vector<int> storeType(1115,0);// competetion
vector<int> assortment(1115,0);// competetion
vector<int> competetionDis(1115,0);// competetion
for(int i = 0; i < 1115; ++i){
//	cout << i << endl;
 	int month = table_train[i][2];
 	int day = table_train[i][3];
 	int year = table_train[i][4];
 	//cout << month << " " << day << " " << year << endl;
 	int monthComp = table_store[i][4];
 	int yearComp = table_store[i][5];
//cout << monthComp <<" " << yearComp << endl;
 	if(year > yearComp || (year==yearComp && month >= monthComp))
 		inComp[i] = 1;
 	if(yearComp == -1 || monthComp == -1)
 		inComp[i] = 0;
 	if(table_store[i][6] == 1){
 		int dayPromo = table_store[i][7]%4 * 7;
 		int monthPromo = table_store[i][7]/4;
 		int yearPromo = table_store[i][8];
//cout << table_store[i][7]<<" " << table_store[i][8] << endl;
 		if(year > yearPromo || (year == yearPromo && month > monthPromo) 
 			|| (year == yearPromo && month == monthPromo && day >= dayPromo)){
 			int month1,month2,month3,month4;
 			month1 = table_store[i][9];
 			month2 = table_store[i][10];
 			month3 = table_store[i][11];
 			month4 = table_store[i][12];
 	//		cout << month1<<" "<< month2 <<" " << month3 <<" "<< month4<<endl;
 			if(month == month1 || month == month2 || month == month3 || month == month4)
 				inPromo[i] = 1;
 		}
 	}

 	storeType[i] = (table_store[i][1]);
 	assortment[i] = (table_store[i][2]);
 	competetionDis[i] =(table_store[i][3]);
 	//cout << storeType[i]<< endl;
 	//if(i%10 == 0) cout <<endl;
}
//cout <<endl;


// adding new features, store type, assortment, competetion distance, inComp, inPromo
vector<vector<int> > mergeTable_Train;
mergeTable_Train.resize(trainN);
for (int i = 0; i < trainN; ++i)
    mergeTable_Train[i].resize(table_train[0].size()+5);

for (int i = 0; i < trainN; ++i){
	for (int j = 0; j < table_train[0].size(); ++j){
		 mergeTable_Train[i][j] = table_train[i][j]; 
	}
	 mergeTable_Train[i][table_train[0].size()] = storeType[(table_train[i][0]-1)%1115];
	 mergeTable_Train[i][table_train[0].size()+1] = assortment[(table_train[i][0]-1)%1115];
	 mergeTable_Train[i][table_train[0].size()+2] = competetionDis[(table_train[i][0]-1)%1115];
	 mergeTable_Train[i][table_train[0].size()+3] = inComp[(table_train[i][0]-1)%1115];
	 mergeTable_Train[i][table_train[0].size()+4] = inPromo[(table_train[i][0]-1)%1115];

}
	

for (int i = 0; i < trainN; ++i){
	for (int j = 0; j < table_train[0].size()+5; ++j){
		cout <<  mergeTable_Train[i][j] << " ";
	}
	cout << endl;
}

	return 0;

}