/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

// Random engine to be used for various methods
static default_random_engine gen;

bool debug = false;

// Particle filter initialization
// Set number of particles and initialize them to first position based on GPS coordinates
void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	num_particles = 100;

    // Create normal (Gaussian) distribution for p_x, p_y, and p_theta
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	for (int i = 0; i < num_particles; i++) {
        Particle p;
		p.id = i;
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
		p.weight = 1.0;

        // Add the particle to the particle filter set
		particles.push_back(p);
        if (debug) {
            cout << "Particle idx:" << i << "; x:" << p.x << "; y:" << p.y<< "; theta:" << theta << endl;
        }
	}

	is_initialized = true;


    if (debug) {
        cout << "-----initialized-----end-----" << endl;
    }

}

void ParticleFilter::prediction(double delta_t, double sigma_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

    for (int i = 0; i < num_particles; i++) {

        // Collect old values
        double x_old = particles[i].x;
        double y_old = particles[i].y;
        double theta_old = particles[i].theta;

        double x_new, y_new, theta_new;

        // Calculate new state
        if (abs(yaw_rate) < 0.00001) {
            // Apply equations of motion model (straight)
            x_new = x_old + velocity * delta_t * cos(theta_old);
            y_new = y_old + velocity * delta_t * sin(theta_old);
            theta_new = theta_old;
        }
        else {
            // Apply equations of motion model (turning)
            x_new = x_old + velocity / yaw_rate * (sin(theta_new) - sin(theta_old));
            y_new = y_old + velocity / yaw_rate * (cos(theta_old) - cos(theta_new));
            theta_new = theta_old + yaw_rate * delta_t;
        }

        // Initialize normal distributions centered around predicted values
        normal_distribution<double> dist_x(x_new, sigma_pos[0]);
        normal_distribution<double> dist_y(y_new, sigma_pos[1]);
        normal_distribution<double> dist_theta(theta_new, sigma_pos[2]);

        // Add noise
        particles[i].x = dist_x(gen);
        particles[i].y = dist_y(gen);
        particles[i].theta = dist_theta(gen);

        if (debug) {
            cout << "predicted particle idx:" << i << "; x:" << particles[i].x << "; y:" << particles[i].y << "; theta:" << particles[i].theta << endl;
        }

    }
    if (debug) {
        cout << "--------predicted------end-----" << endl;
    }

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

    for (auto& obs : observations) {
        double min_dist = numeric_limits<double>::max();

        for (const auto& pred_obs : predicted) {
            double d = dist(obs.x, obs.y, pred_obs.x, pred_obs.y);
            if (d < min_dist) {
                obs.id = pred_obs.id;
                min_dist = d;
            }
        }
    }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
    // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
    //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
    //   according to the MAP'S coordinate system. You will need to transform between the two systems.
    //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
    //   The following is a good resource for the theory:
    //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
    //   and the following is a good resource for the actual equation to implement (look at equation
    //   3.33
    //   http://planning.cs.uiuc.edu/node99.html


    // Gather std values for readability
    double std_x = std_landmark[0];
    double std_y = std_landmark[1];
    weights.clear();

    // For each particle
    for (int i = 0; i < num_particles; i++) {

        // Gather current particle values
        double par_x = particles[i].x;
        double par_y = particles[i].y;
        double par_theta = particles[i].theta;

        // Create a vector to hold all landmarks within sensor range
        vector<LandmarkObs> predicted_landmarks;

        // For each map landmark
        for (const auto &map_landmark : map_landmarks.landmark_list) {

            // Get id and x, y coordinates
            double landmark_x = (double) map_landmark.x_f;
            double landmark_y = (double) map_landmark.y_f;
            int landmark_id = map_landmark.id_i;

            double d = dist(par_x, par_y, landmark_x, landmark_y);
            if (d < sensor_range) {
                LandmarkObs landmark_pred;
                landmark_pred.x = landmark_x;
                landmark_pred.y = landmark_y;
                landmark_pred.id = landmark_id;
                predicted_landmarks.push_back(landmark_pred);
            }
        }
        // Create and fill a copy of the list of observations transformed from vehicle coordinates to map coordinates
        vector<LandmarkObs> observed_landmarks_map;

        for (int j = 0; j < observations.size(); j++) {
            // Convert observation from particle (vehicle) to map coordinate system
            LandmarkObs translated_obs;
            translated_obs.x = cos(par_theta) * observations[j].x - sin(par_theta) * observations[j].y + par_x;
            translated_obs.y = sin(par_theta) * observations[j].x + cos(par_theta) * observations[j].y + par_y;

            observed_landmarks_map.push_back(translated_obs);
        }

        // Perform dataAssociation for the predictions and transformed observations on current particle
        dataAssociation(predicted_landmarks, observed_landmarks_map);

        // Calculate likelihood of current observations
        double particle_likelihood = 1.0;

        double mu_x, mu_y;

        for (const auto &obs : observed_landmarks_map) {

            // Find corresponding landmark on map for centering Gaussian distribution
            for (const auto &land : predicted_landmarks) {
                if (obs.id == land.id) {
                    mu_x = land.x;
                    mu_y = land.y;
                    break;
                }
            }

            double normalize = 2 * M_PI * std_x * std_y;
            double prob = exp(
                    -(pow(obs.x - mu_x, 2) / (2 * std_x * std_x) + pow(obs.y - mu_y, 2) / (2 * std_y * std_y)));

            particle_likelihood *= prob / normalize;
        }

        particles[i].weight = particle_likelihood;

    } // End of loop for all particles

    // Calculate weight normalization factor
    double normalize = 0.0;
    for (const auto &particle : particles) {
        normalize += particle.weight;
    }

    // Normalize weights so they sum to one
    for (auto &particle : particles) {
        particle.weight /= (normalize + numeric_limits<double>::epsilon());
    }

    if (debug) {
        cout << "--------updated weights-------end------" << endl;
    }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution


    vector<double> particle_weights;
    for (const auto& particle : particles) {
        particle_weights.push_back(particle.weight);
    }

    discrete_distribution<int> weighted_distribution(particle_weights.begin(), particle_weights.end());

    vector<Particle> resampled_particles;
    for (int i = 0; i < num_particles; i++) {
        int k = weighted_distribution(gen);
        resampled_particles.push_back(particles[k]);
    }

    particles = resampled_particles;

    // Reset weights for all particles
    for (auto& particle : particles) {

        particle.weight = 1.0;
    }

    if (debug) {
        cout << "--------resampled-------end------" << endl;
    }
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y) {
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best) {
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
