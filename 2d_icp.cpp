#include <iostream>
#include <vector>
#include <string>
#include <iterator>
#include <random>
#include <bits/stdc++.h>

#include <Eigen/Core>
#include <sophus/so3.h>

using namespace std;
using namespace Eigen;

void CreateData(vector<Vector2d>& p1, vector<Vector2d>& p2)
{
    Vector2d p_1(0, 0);
    p1.push_back(p_1);
    Vector2d p_2(2, 0);
    p1.push_back(p_2);
    Vector2d p_3(2, 1);
    p1.push_back(p_3);
    Vector2d p_4(0, 1);
    p1.push_back(p_4);
    Vector2d p_5(0, 5);
    p1.push_back(p_5);
    Vector2d p_6(2, 5);
    p1.push_back(p_6);
    Vector2d p_7(3, 7);
    p1.push_back(p_7);
    Vector2d p_8(9, 4);
    p1.push_back(p_8);

    Matrix2d R;
    double angle = 20 * 3.14 / 180;  // 旋转 10 度
    R(0, 0) = cos(angle);
    R(0, 1) = -sin(angle);
    R(1, 0) = sin(angle);
    R(1, 1) = cos(angle);

    Vector2d t(0.5, 0.5);
    cout<< "init R:" <<endl<< R <<endl;
    cout<< "init t:" <<endl<< t.transpose() <<endl;

    // Define random generator with Gaussian distribution
    const double mean = 0.0;//均值
    const double stddev = 0.1;//标准差
    std::default_random_engine generator;
    std::normal_distribution<double> dist(mean, stddev);

    for (int i=0;i<p1.size();i++) {
        Vector2d p_ = R * p1[i] + t;
        p_[0] += dist(generator);
        p_[1] += dist(generator);
        p2.push_back(p_);
    }
    cout<< "create data down..." <<endl;
}

void printData(vector<Vector2d>& p1, vector<Vector2d>& p2)
{
    cout<< "----- print data -----" <<endl;
    cout<< "p1 data: " <<endl;
    for (int i=0;i<p1.size();i++) {
        cout<< p1[i].transpose() <<endl;
    }

    cout<< "p2 data: " <<endl;
    for (int i=0;i<p2.size();i++) {
        cout<< p2[i].transpose() <<endl;
    }
}

void MatchPoint(vector<Vector2d>& p1, vector<Vector2d>& p2, vector<int>& match, Matrix2d& R, Vector2d& t)
{
    for (int i=0;i < p2.size();i++) {
        Vector2d p_ = R.inverse() * p2[i] - t;
        double dis = 1000;
        int id;
        for (int j=0;j<p1.size();j++) {
            double dis_ = sqrt(pow(p_[0] - p1[j][0], 2) + pow(p_[1] - p1[j][1], 2));
            if (dis_ < dis) {
                id = j;
                dis = dis_;
            }
        }
        match.push_back(id);
    }
}

int main(int argv, char** argc)
{
    vector< Vector2d > p1;
    vector< Vector2d > p2;

    CreateData(p1, p2);
    // printData(p1, p2);

    Vector2d p1_mean;
    Vector2d p2_mean;
    for (int i=0;i < p1.size();i++) {
        p1_mean += p1[i];
    }
    p1_mean /= p1.size();

    for (int i=0;i < p2.size();i++) {
        p2_mean += p2[i];
    }
    p2_mean /= p2.size();

    vector< Vector2d > p1_norm;
    vector< Vector2d > p2_norm;
    for (int i=0;i < p1.size();i++) {
        Vector2d p_ = p1[i] - p1_mean;
        p1_norm.push_back(p_);
    }
    for (int i=0;i < p2.size();i++) {
        Vector2d p_ = p2[i] - p2_mean;
        p2_norm.push_back(p_);
    }

    Matrix2d R = Matrix2d::Identity();
    Vector2d t = Vector2d::Zero();

    for (int iter=0;iter < 10;iter++) {
        cout<< "---- iter: " << iter << " -----" <<endl;

        vector<int> match_index;
        MatchPoint(p1_norm, p2_norm, match_index, R, t);
        // for (int i=0;i<match_index.size();i++) {
        //     cout<< match_index[i] << " ";
        // }

        Matrix2d W = Matrix2d::Zero();
        for (int i=0;i<p2_norm.size();i++) {
            Vector2d p_ = p1_norm[match_index[i]];
            W += p2_norm[i] * p_.transpose();
        }
        // cout<< "W:" <<endl<< W <<endl;

        JacobiSVD<Matrix2d> svd(W, ComputeFullU | ComputeFullV);
        Matrix2d U = svd.matrixU();
        Matrix2d V = svd.matrixV();

        R = U * V.transpose();
        t = p2_mean - R * p1_mean;
        // cout<< "R:" <<endl<< R <<endl;
        // cout<< "t:" <<endl<< t.transpose() <<endl;

        Vector2d error(0, 0);
        for (int i=0;i< p2.size();i++) {
            error += R * p1[match_index[i]] + t - p2[i];
        }
        error /= p2.size();
        double error_ = sqrt(error[0] * error[0] + error[1] * error[1]);
        // cout<< "error: " << error_ <<endl;

        if (error_ < 0.001) {
            cout<< "error is small, break" <<endl;
            break;
        }
    }

    return 0;
}