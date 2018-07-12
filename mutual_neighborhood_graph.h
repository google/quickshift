/*
Copyright 2018 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include <map>
#include <stack>
#include <set>
#include <vector>
#include <algorithm>
#include <string>
#include <stdlib.h>
#include <fstream>
#include <math.h>
using namespace std;

struct Node {
    /*
    Node struct for our k-NN or neighborhood Graph
    */
    int index;
    int rank;
    Node * parent;
    set <Node * > children;
    Node(int idx) {
    	index = idx;
    	rank = 0;
    	parent = NULL;
    	children.clear();
    }

};



struct Graph {
    /*
    Graph struct.
    Allows us to build the graph one node at a time
    */
    vector <Node *> nodes;
    map <int, Node * > M;
    set <Node * > intersecting_sets;
    Graph() {
        M.clear();
        intersecting_sets.clear();
        nodes.clear();
    }

    Node * get_root(Node * node) {

         if (node->parent != NULL) {
         	node->parent->children.erase(node);
         	node->parent = get_root(node->parent);
         	node->parent->children.insert(node);
         	return node->parent;
         } else {
            return node;
         }    	
    }

    void add_node(int idx) {
       nodes.push_back(new Node(idx));
       M[idx] = nodes[nodes.size() - 1];
    }

    void add_edge(int n1, int n2) {
    	Node * r1 = get_root(M[n1]);
    	Node * r2 = get_root(M[n2]);
    	if (r1 != r2) {
    		if (r1->rank > r2->rank) {
    			r2->parent = r1;
    			r1->children.insert(r2);
    			if (intersecting_sets.count(r2)) {
    				intersecting_sets.erase(r2);
    				intersecting_sets.insert(r1);
    			}
    		} else {
    			r1->parent = r2;
    			r2->children.insert(r1);
    			if (intersecting_sets.count(r1)) {
    				intersecting_sets.erase(r1);
    				intersecting_sets.insert(r2);
    			}

    			if (r1->rank == r2->rank) {
    				r2->rank++;
    			}
    		}
    	}
    }

    vector <int> get_connected_component(int n) {
        Node * r = get_root(M[n]);
        vector <int> L;
        stack <Node * > s;
        s.push(r);
        while (!s.empty()) {
            Node * top = s.top(); s.pop();
            L.push_back(top->index);
            for (set<Node * >::iterator it = top->children.begin();
            	                    it != top->children.end();
            	                    ++it) {
                s.push(*it);
    		}
    	}
    	return L;
    }


    bool component_seen(int n) {
        Node * r = get_root(M[n]);
        if (intersecting_sets.count(r)) {
             return true;
        }
        intersecting_sets.insert(r);
        return false;
    }

    int GET_ROOT(int idx) {
    	Node * r = get_root(M[idx]);
    	return r->index;
    }

    vector <int> GET_CHILDREN(int idx) {
    	Node * r = M[idx];
    	vector <int> to_ret;
    	for (set<Node *>::iterator it = r->children.begin();
    		                       it != r->children.end();
    		                       ++it) {
    		to_ret.push_back((*it)->index);
    	}
    	return to_ret;
    }

};


struct NodeBasic {
    int index;
    int rank;
    NodeBasic * parent;
    NodeBasic(int idx) {
        index = idx;
        rank = 0;
        parent = NULL;
    }
};

struct GraphBasic {
    /*
    Basic disjoint set data structure.    */
    vector<NodeBasic *> M;
    GraphBasic(const int n) {
        M.clear();
        for (int i = 0; i < n; ++i) {
            M.push_back(new NodeBasic(i));
        }
    }

    NodeBasic * get_root(NodeBasic * node) {
         if (!node) return NULL;
         if (!node->parent) return node;
        node->parent = get_root(node->parent);
        return node->parent;     
    }

    void add_edge(const int n1, const int n2) {
        NodeBasic * r1 = get_root(M[n1]);
        NodeBasic * r2 = get_root(M[n2]);
        if (!r1 || !r2) return;
        if (r1 != r2) {
            if (r1->rank > r2->rank) {
                r2->parent = r1;
            } else {
                r1->parent = r2;
                if (r1->rank == r2->rank) {
                    r2->rank++;
                }
            }
        }
    }
};


void compute_mutual_knn(int n, int k,
                    int d,
                    double * radii,
                    int * neighbors,
                    double beta,
                    double epsilon,
                    int * result) {
    /* Given the kNN density and neighbors
        We build the k-NN graph / cluster tree and return the estimated modes.
        Note that here, we don't require the dimension of the dataset 
        Returns array of estimated mode membership, where each index cosrresponds
        the respective index in the density array. Points without
        membership are assigned -1 */

    vector<pair <double, int> > knn_radii(n);
    vector <set <int> > knn_neighbors(n);



    for (int i = 0; i < n; ++i) {
        knn_radii[i].first = radii[i];
        knn_radii[i].second = i;

        for (int j = 0; j < k; ++j) {
            knn_neighbors[i].insert(neighbors[i * k + j]);
        }
    }

    int m_hat[n];
    int cluster_membership[n];
    int n_chosen_points = 0;
    int n_chosen_clusters = 0;
    sort(knn_radii.begin(), knn_radii.end());

    Graph G = Graph();

    int last_considered = 0;
    int last_pruned = 0;
    
    for (int i = 0; i < n; ++i) {
        while (last_pruned < n && pow(1. + epsilon, 1. / d) * knn_radii[i].first > knn_radii[last_pruned].first) { 

            G.add_node(knn_radii[last_pruned].second);

            for (set <int>::iterator it = knn_neighbors[knn_radii[last_pruned].second].begin();
                                    it != knn_neighbors[knn_radii[last_pruned].second].end();
                                 ++it) {
                if (G.M.count(*it)) {
                  if (knn_neighbors[*it].count(knn_radii[last_pruned].second)) {
                        G.add_edge(knn_radii[last_pruned].second, *it);
                    }
             
                }

            }
            last_pruned++;
        }


        while(knn_radii[i].first * pow(1. - beta, 1. / d) > knn_radii[last_considered].first) {

            if (!G.component_seen(knn_radii[last_considered].second)) {
                vector <int> res = G.get_connected_component(knn_radii[last_considered].second);
                for (size_t j = 0; j < res.size(); j++) {
                    if (radii[res[j]] <= knn_radii[i].first) {
                        cluster_membership[n_chosen_points] = n_chosen_clusters;
                        m_hat[n_chosen_points++] = res[j];
                    }

                }
                n_chosen_clusters++;
            }
            last_considered++;
        }
    }

    for (int i = 0; i < n; ++i) {
        result[i] = -1;
    }
    
    for (int i = 0; i < n_chosen_points; ++i) {
        result[m_hat[i]] = cluster_membership[i];
    }


}

double dist(int i, int j, int d, double ** dataset) {
    double sum = 0.;
    for (int m = 0; m < d; ++m) {
        sum += (dataset[i][m] - dataset[j][m]) * (dataset[i][m] - dataset[j][m]);
    }
    return sum;
}

void cluster_remaining(
    int n, int k, int d,
    double * dataset,
     double * radii,
     int * neighbors,
     int * initial_memberships,
     int * result) {

    int ** knn_neighbors = new int*[n];
    double ** data;
    data = new double *[n];
    for (int i = 0; i < n; ++i) {
        data[i] = new double[d];
        knn_neighbors[i] = new int[k];
    }
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < k; ++j) {
            knn_neighbors[i][j] = neighbors[i * k + j];
        }
    }
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < d; ++j) {
            data[i][j] = dataset[i * d + j];
        }
    }

    // Final clusters.
    GraphBasic H = GraphBasic(n);

    int n_chosen_clusters = 0;
    for (int i = 0; i < n; ++i) {
        if (n_chosen_clusters < initial_memberships[i]) {
            n_chosen_clusters = initial_memberships[i];
        }
    }
    n_chosen_clusters += 1;
    vector<vector<int> > modal_sets(n_chosen_clusters);
    for (int c = 0; c < n_chosen_clusters; ++c) {
        modal_sets.push_back(vector<int>());
    }
    for (int i = 0; i < n; ++i) {
        if (initial_memberships[i] >= 0) {
            modal_sets[initial_memberships[i]].push_back(i);
        }
    }
    for (int c = 0; c < n_chosen_clusters; ++c) {
        for (size_t i = 0; i < modal_sets[c].size() - 1; ++i) {
            H.add_edge(modal_sets[c][i], modal_sets[c][i+1]);
        }
    }
    int next = -1;
    double dt, best_distance = 0.;
    for (int i = 0; i < n; ++i) {
        if (initial_memberships[i] >= 0) {
            continue;
        }
        next = -1;
        for (int j = 0; j < k; ++j) {
            if (radii[knn_neighbors[i][j]] < radii[i]) {
                next = knn_neighbors[i][j];
                break;
            }
        }

        if (next < 0) {
            best_distance = 1000000000.;
            for (int j = 0; j < n; ++j) {
                if (radii[j] >= radii[i]) {
                    continue;
                }
                dt = 0.0;
                for (int m = 0; m < d; ++m) {
                    dt += (data[i][m] - data[j][m]) * (data[i][m] - data[j][m]);
                }
                if (best_distance > dt) {
                    best_distance = dt;
                    next = j;
                }
            }
        }
        H.add_edge(i, next);
    }
    for (int i = 0; i < n; ++i) {
        result[i] = -1;
    }
    int n_clusters = 0;
    map<int, int> label_mapping;
    for (int i = 0; i < n; ++i) {
        if (result[i] < 0) {
            int label = (H.get_root(H.M[i]))->index;
            if (label_mapping.count(label)) {
                result[i] = label_mapping[label];
            } else {
                label_mapping[label] = n_clusters;
                result[i] = n_clusters;
                n_clusters++;
            }
        }
    }
}
