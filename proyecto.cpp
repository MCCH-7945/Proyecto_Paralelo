#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <random>
#include <limits>
#include <cmath>
#include <stdexcept>
#include <iomanip>
#include <chrono>
#include <set>
#include <algorithm>

using namespace std;

struct Point {
    vector<double> values;
    int cluster = -1;
};

struct KMeansResult {
    vector<vector<double>> centroids;
    int iterations;
};

struct RunMetrics {
    double time_seconds = 0.0;
    double wcss = 0.0;
    int iterations = 0;
};

struct ParallelComparisonMetrics {
    double serial_time = 0.0;
    double parallel_time = 0.0;
    double speedup = 0.0;
    double efficiency = 0.0;
    double improvement_percent = 0.0;
    double parallel_cost = 0.0;
    double serial_wcss = 0.0;
    double parallel_wcss = 0.0;
    double serial_iterations = 0.0;
    double parallel_iterations = 0.0;
};

vector<string> split(const string& line, char delimiter) {
    vector<string> tokens;
    string token;
    stringstream ss(line);

    while (getline(ss, token, delimiter)) {
        tokens.push_back(token);
    }

    return tokens;
}

vector<Point> readCSV(const string& filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        throw runtime_error("No se pudo abrir el archivo: " + filename);
    }

    vector<Point> points;
    string line;
    size_t expected_dim = 0;

    while (getline(file, line)) {
        if (line.empty()) continue;

        vector<string> tokens = split(line, ',');
        vector<double> coords;

        for (const string& token : tokens) {
            if (!token.empty()) {
                coords.push_back(stod(token));
            }
        }

        if (coords.empty()) continue;

        if (expected_dim == 0) {
            expected_dim = coords.size();
            if (expected_dim != 2 && expected_dim != 3) {
                throw runtime_error("El archivo debe contener puntos en 2D o 3D.");
            }
        } else if (coords.size() != expected_dim) {
            throw runtime_error("Todas las filas del CSV deben tener la misma dimension.");
        }

        points.push_back({coords, -1});
    }

    file.close();

    if (points.empty()) {
        throw runtime_error("El archivo CSV no contiene puntos validos.");
    }

    return points;
}

void writeAssignmentsCSV(const string& filename, const vector<Point>& points) {
    ofstream file(filename);
    if (!file.is_open()) {
        throw runtime_error("No se pudo crear el archivo: " + filename);
    }

    for (const auto& p : points) {
        for (size_t i = 0; i < p.values.size(); ++i) {
            file << p.values[i] << ",";
        }
        file << p.cluster << "\n";
    }

    file.close();
}

double squaredDistance(const vector<double>& a, const vector<double>& b) {
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

void resetClusters(vector<Point>& points) {
    for (auto& p : points) {
        p.cluster = -1;
    }
}

vector<vector<double>> initializeCentroids(const vector<Point>& points, int k, unsigned int seed = 42) {
    if (k <= 0 || k > static_cast<int>(points.size())) {
        throw runtime_error("k debe ser mayor que 0 y menor o igual al numero de puntos.");
    }

    vector<vector<double>> centroids;
    vector<int> chosen_indices;
    mt19937 gen(seed);
    uniform_int_distribution<int> dist(0, static_cast<int>(points.size()) - 1);

    while (static_cast<int>(centroids.size()) < k) {
        int idx = dist(gen);

        bool already_used = false;
        for (int used : chosen_indices) {
            if (used == idx) {
                already_used = true;
                break;
            }
        }

        if (!already_used) {
            chosen_indices.push_back(idx);
            centroids.push_back(points[idx].values);
        }
    }

    return centroids;
}

int assignClustersSerial(vector<Point>& points, const vector<vector<double>>& centroids) {
    int changes = 0;

    for (auto& p : points) {
        double best_dist = numeric_limits<double>::max();
        int best_cluster = -1;

        for (size_t c = 0; c < centroids.size(); ++c) {
            double dist = squaredDistance(p.values, centroids[c]);
            if (dist < best_dist) {
                best_dist = dist;
                best_cluster = static_cast<int>(c);
            }
        }

        if (p.cluster != best_cluster) {
            p.cluster = best_cluster;
            changes++;
        }
    }

    return changes;
}

int assignClustersParallel(vector<Point>& points, const vector<vector<double>>& centroids, int num_threads) {
    int changes = 0;
    int n = static_cast<int>(points.size());

    #pragma omp parallel for num_threads(num_threads) reduction(+:changes) schedule(static)
    for (int i = 0; i < n; ++i) {
        double best_dist = numeric_limits<double>::max();
        int best_cluster = -1;

        for (size_t c = 0; c < centroids.size(); ++c) {
            double dist = squaredDistance(points[i].values, centroids[c]);
            if (dist < best_dist) {
                best_dist = dist;
                best_cluster = static_cast<int>(c);
            }
        }

        if (points[i].cluster != best_cluster) {
            points[i].cluster = best_cluster;
            changes++;
        }
    }

    return changes;
}

vector<vector<double>> recomputeCentroidsSerial(const vector<Point>& points, int k, size_t dim, unsigned int seed = 42) {
    vector<vector<double>> new_centroids(k, vector<double>(dim, 0.0));
    vector<int> counts(k, 0);

    for (const auto& p : points) {
        int c = p.cluster;
        if (c < 0 || c >= k) continue;

        for (size_t d = 0; d < dim; ++d) {
            new_centroids[c][d] += p.values[d];
        }
        counts[c]++;
    }

    mt19937 gen(seed);
    uniform_int_distribution<int> dist(0, static_cast<int>(points.size()) - 1);

    for (int c = 0; c < k; ++c) {
        if (counts[c] == 0) {
            int random_index = dist(gen);
            new_centroids[c] = points[random_index].values;
        } else {
            for (size_t d = 0; d < dim; ++d) {
                new_centroids[c][d] /= counts[c];
            }
        }
    }

    return new_centroids;
}

vector<vector<double>> recomputeCentroidsParallel(const vector<Point>& points, int k, size_t dim, unsigned int seed, int num_threads) {
    vector<vector<double>> global_sums(k, vector<double>(dim, 0.0));
    vector<int> global_counts(k, 0);

    #pragma omp parallel num_threads(num_threads)
    {
        vector<vector<double>> local_sums(k, vector<double>(dim, 0.0));
        vector<int> local_counts(k, 0);

        #pragma omp for schedule(static)
        for (int i = 0; i < static_cast<int>(points.size()); ++i) {
            int c = points[i].cluster;
            if (c < 0 || c >= k) continue;

            for (size_t d = 0; d < dim; ++d) {
                local_sums[c][d] += points[i].values[d];
            }
            local_counts[c]++;
        }

        #pragma omp critical
        {
            for (int c = 0; c < k; ++c) {
                for (size_t d = 0; d < dim; ++d) {
                    global_sums[c][d] += local_sums[c][d];
                }
                global_counts[c] += local_counts[c];
            }
        }
    }

    vector<vector<double>> new_centroids(k, vector<double>(dim, 0.0));

    mt19937 gen(seed);
    uniform_int_distribution<int> dist(0, static_cast<int>(points.size()) - 1);

    for (int c = 0; c < k; ++c) {
        if (global_counts[c] == 0) {
            int random_index = dist(gen);
            new_centroids[c] = points[random_index].values;
        } else {
            for (size_t d = 0; d < dim; ++d) {
                new_centroids[c][d] = global_sums[c][d] / global_counts[c];
            }
        }
    }

    return new_centroids;
}

KMeansResult kmeansSerial(vector<Point>& points, int k, int max_iters = 100, double tol = 1e-6, unsigned int seed = 42) {
    if (points.empty()) {
        throw runtime_error("No hay puntos para clusterizar.");
    }

    size_t dim = points[0].values.size();
    vector<vector<double>> centroids = initializeCentroids(points, k, seed);

    int iterations = 0;

    for (int iter = 0; iter < max_iters; ++iter) {
        iterations++;

        int changes = assignClustersSerial(points, centroids);
        vector<vector<double>> new_centroids = recomputeCentroidsSerial(points, k, dim, seed + iter + 1);

        double max_shift = 0.0;
        for (int c = 0; c < k; ++c) {
            double shift = sqrt(squaredDistance(centroids[c], new_centroids[c]));
            if (shift > max_shift) {
                max_shift = shift;
            }
        }

        centroids = new_centroids;

        if (changes == 0 || max_shift < tol) {
            break;
        }
    }

    return {centroids, iterations};
}

KMeansResult kmeansParallel(vector<Point>& points, int k, int num_threads, int max_iters = 100, double tol = 1e-6, unsigned int seed = 42) {
    if (points.empty()) {
        throw runtime_error("No hay puntos para clusterizar.");
    }

    size_t dim = points[0].values.size();
    vector<vector<double>> centroids = initializeCentroids(points, k, seed);

    int iterations = 0;

    for (int iter = 0; iter < max_iters; ++iter) {
        iterations++;

        int changes = assignClustersParallel(points, centroids, num_threads);
        vector<vector<double>> new_centroids = recomputeCentroidsParallel(points, k, dim, seed + iter + 1, num_threads);

        double max_shift = 0.0;
        for (int c = 0; c < k; ++c) {
            double shift = sqrt(squaredDistance(centroids[c], new_centroids[c]));
            if (shift > max_shift) {
                max_shift = shift;
            }
        }

        centroids = new_centroids;

        if (changes == 0 || max_shift < tol) {
            break;
        }
    }

    return {centroids, iterations};
}

double computeWCSS(const vector<Point>& points, const vector<vector<double>>& centroids) {
    double wcss = 0.0;

    for (const auto& p : points) {
        if (p.cluster >= 0) {
            wcss += squaredDistance(p.values, centroids[p.cluster]);
        }
    }

    return wcss;
}

vector<int> computeClusterSizes(const vector<Point>& points, int k) {
    vector<int> sizes(k, 0);

    for (const auto& p : points) {
        if (p.cluster >= 0 && p.cluster < k) {
            sizes[p.cluster]++;
        }
    }

    return sizes;
}

void printCentroids(const vector<vector<double>>& centroids) {
    cout << "\nCentroides finales:\n";
    for (size_t i = 0; i < centroids.size(); ++i) {
        cout << "Cluster " << i << ": ";
        for (size_t d = 0; d < centroids[i].size(); ++d) {
            cout << fixed << setprecision(6) << centroids[i][d];
            if (d + 1 < centroids[i].size()) cout << ", ";
        }
        cout << "\n";
    }
}

void printClusterSizes(const vector<int>& cluster_sizes) {
    cout << "\nTamano de clusters:\n";
    for (size_t i = 0; i < cluster_sizes.size(); ++i) {
        cout << "Cluster " << i << ": " << cluster_sizes[i] << " puntos\n";
    }
}

RunMetrics runSerialOnce(const vector<Point>& original_points, int k, int max_iters, double tol, unsigned int seed) {
    vector<Point> points = original_points;
    resetClusters(points);

    auto start = chrono::high_resolution_clock::now();
    KMeansResult result = kmeansSerial(points, k, max_iters, tol, seed);
    auto end = chrono::high_resolution_clock::now();

    RunMetrics metrics;
    metrics.time_seconds = chrono::duration<double>(end - start).count();
    metrics.wcss = computeWCSS(points, result.centroids);
    metrics.iterations = result.iterations;
    return metrics;
}

RunMetrics runParallelOnce(const vector<Point>& original_points, int k, int num_threads, int max_iters, double tol, unsigned int seed) {
    vector<Point> points = original_points;
    resetClusters(points);

    auto start = chrono::high_resolution_clock::now();
    KMeansResult result = kmeansParallel(points, k, num_threads, max_iters, tol, seed);
    auto end = chrono::high_resolution_clock::now();

    RunMetrics metrics;
    metrics.time_seconds = chrono::duration<double>(end - start).count();
    metrics.wcss = computeWCSS(points, result.centroids);
    metrics.iterations = result.iterations;
    return metrics;
}

ParallelComparisonMetrics averageComparisonMetrics(
    const vector<Point>& original_points,
    int k,
    int num_threads,
    int max_iters,
    double tol,
    int repetitions,
    unsigned int base_seed
) {
    ParallelComparisonMetrics avg;

    for (int rep = 0; rep < repetitions; ++rep) {
        unsigned int seed = base_seed + static_cast<unsigned int>(rep * 100);

        RunMetrics serial_metrics = runSerialOnce(original_points, k, max_iters, tol, seed);
        RunMetrics parallel_metrics = runParallelOnce(original_points, k, num_threads, max_iters, tol, seed);

        double speedup = serial_metrics.time_seconds / parallel_metrics.time_seconds;
        double efficiency = speedup / num_threads;
        double improvement_percent = ((serial_metrics.time_seconds - parallel_metrics.time_seconds) / serial_metrics.time_seconds) * 100.0;
        double parallel_cost = num_threads * parallel_metrics.time_seconds;

        avg.serial_time += serial_metrics.time_seconds;
        avg.parallel_time += parallel_metrics.time_seconds;
        avg.speedup += speedup;
        avg.efficiency += efficiency;
        avg.improvement_percent += improvement_percent;
        avg.parallel_cost += parallel_cost;
        avg.serial_wcss += serial_metrics.wcss;
        avg.parallel_wcss += parallel_metrics.wcss;
        avg.serial_iterations += serial_metrics.iterations;
        avg.parallel_iterations += parallel_metrics.iterations;
    }

    avg.serial_time /= repetitions;
    avg.parallel_time /= repetitions;
    avg.speedup /= repetitions;
    avg.efficiency /= repetitions;
    avg.improvement_percent /= repetitions;
    avg.parallel_cost /= repetitions;
    avg.serial_wcss /= repetitions;
    avg.parallel_wcss /= repetitions;
    avg.serial_iterations /= repetitions;
    avg.parallel_iterations /= repetitions;

    return avg;
}

vector<int> getRequiredThreadConfigs() {
    int virtual_cores = omp_get_num_procs();
    if (virtual_cores <= 0) {
        virtual_cores = 1;
    }

    set<int> unique_threads;
    unique_threads.insert(1);
    unique_threads.insert(max(1, virtual_cores / 2));
    unique_threads.insert(max(1, virtual_cores));
    unique_threads.insert(max(1, 2 * virtual_cores));

    vector<int> thread_configs(unique_threads.begin(), unique_threads.end());
    sort(thread_configs.begin(), thread_configs.end());
    return thread_configs;
}

void writeExperimentHeaderIfNeeded(const string& metrics_file) {
    ifstream infile(metrics_file);
    bool exists_and_not_empty = infile.good() && infile.peek() != ifstream::traits_type::eof();
    infile.close();

    if (!exists_and_not_empty) {
        ofstream out(metrics_file);
        if (!out.is_open()) {
            throw runtime_error("No se pudo crear el archivo de metricas: " + metrics_file);
        }

        out << "dataset"
            << ",n_points"
            << ",dimension"
            << ",k"
            << ",repetitions"
            << ",threads"
            << ",avg_time_serial"
            << ",avg_time_parallel"
            << ",avg_speedup"
            << ",avg_efficiency"
            << ",avg_improvement_percent"
            << ",avg_parallel_cost"
            << ",avg_wcss_serial"
            << ",avg_wcss_parallel"
            << ",avg_iterations_serial"
            << ",avg_iterations_parallel"
            << "\n";

        out.close();
    }
}

void appendExperimentRow(
    const string& metrics_file,
    const string& dataset_name,
    int n_points,
    int dimension,
    int k,
    int repetitions,
    int threads,
    const ParallelComparisonMetrics& m
) {
    ofstream out(metrics_file, ios::app);
    if (!out.is_open()) {
        throw runtime_error("No se pudo abrir el archivo de metricas: " + metrics_file);
    }

    out << dataset_name << ","
        << n_points << ","
        << dimension << ","
        << k << ","
        << repetitions << ","
        << threads << ","
        << fixed << setprecision(10)
        << m.serial_time << ","
        << m.parallel_time << ","
        << m.speedup << ","
        << m.efficiency << ","
        << m.improvement_percent << ","
        << m.parallel_cost << ","
        << m.serial_wcss << ","
        << m.parallel_wcss << ","
        << m.serial_iterations << ","
        << m.parallel_iterations
        << "\n";

    out.close();
}

string basenameOnly(const string& path) {
    size_t pos1 = path.find_last_of("/\\");
    if (pos1 == string::npos) return path;
    return path.substr(pos1 + 1);
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    try {
        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        if (rank == 0) {
            if (argc < 5) {
                cerr << "Uso:\n";
                cerr << "  " << argv[0] << " <input.csv> <output_or_metrics.csv> <k> <modo> [max_iters] [num_threads_or_repetitions]\n\n";
                cerr << "Modos:\n";
                cerr << "  serial\n";
                cerr << "  parallel\n";
                cerr << "  experiment\n\n";
                cerr << "Ejemplos:\n";
                cerr << "  " << argv[0] << " 98000_data_2d.csv salida_serial.csv 3 serial 100\n";
                cerr << "  " << argv[0] << " 98000_data_3d.csv salida_parallel.csv 4 parallel 100 8\n";
                cerr << "  " << argv[0] << " 98000_data_2d.csv metricas_experimento.csv 3 experiment 100 10\n";
                MPI_Finalize();
                return 1;
            }

            string input_file = argv[1];
            string output_file = argv[2];
            int k = stoi(argv[3]);
            string mode = argv[4];
            int max_iters = (argc >= 6) ? stoi(argv[5]) : 100;
            double tol = 1e-6;
            unsigned int base_seed = 42;

            cout << "Leyendo archivo: " << input_file << "\n";
            vector<Point> original_points = readCSV(input_file);

            int n_points = static_cast<int>(original_points.size());
            int dimension = static_cast<int>(original_points[0].values.size());
            string dataset_name = basenameOnly(input_file);

            cout << "Procesos MPI detectados: " << size << "\n";
            cout << "Numero de puntos: " << n_points << "\n";
            cout << "Dimension detectada: " << dimension << "D\n";
            cout << "Numero de clusters (k): " << k << "\n";
            cout << "Maximo de iteraciones: " << max_iters << "\n";

            if (mode == "serial") {
                cout << "Modo de ejecucion: SERIAL\n";

                vector<Point> points = original_points;
                resetClusters(points);

                auto start = chrono::high_resolution_clock::now();
                KMeansResult result = kmeansSerial(points, k, max_iters, tol, base_seed);
                auto end = chrono::high_resolution_clock::now();

                double time_serial = chrono::duration<double>(end - start).count();
                double wcss = computeWCSS(points, result.centroids);
                vector<int> cluster_sizes = computeClusterSizes(points, k);

                cout << "\n===== RESULTADOS SERIAL =====\n";
                cout << "Iteraciones ejecutadas: " << result.iterations << "\n";
                cout << "Tiempo serial: " << time_serial << " segundos\n";
                cout << "WCSS final: " << fixed << setprecision(6) << wcss << "\n";

                printClusterSizes(cluster_sizes);
                printCentroids(result.centroids);

                writeAssignmentsCSV(output_file, points);
                cout << "\nArchivo de salida generado: " << output_file << "\n";
            }
            else if (mode == "parallel") {
                int num_threads = (argc >= 7) ? stoi(argv[6]) : omp_get_max_threads();

                cout << "Modo de ejecucion: PARALELO (OpenMP)\n";
                cout << "Numero de hilos: " << num_threads << "\n";

                vector<Point> points_serial = original_points;
                vector<Point> points_parallel = original_points;
                resetClusters(points_serial);
                resetClusters(points_parallel);

                auto start_serial = chrono::high_resolution_clock::now();
                KMeansResult result_serial = kmeansSerial(points_serial, k, max_iters, tol, base_seed);
                auto end_serial = chrono::high_resolution_clock::now();

                auto start_parallel = chrono::high_resolution_clock::now();
                KMeansResult result_parallel = kmeansParallel(points_parallel, k, num_threads, max_iters, tol, base_seed);
                auto end_parallel = chrono::high_resolution_clock::now();

                double time_serial = chrono::duration<double>(end_serial - start_serial).count();
                double time_parallel = chrono::duration<double>(end_parallel - start_parallel).count();

                double speedup = time_serial / time_parallel;
                double efficiency = speedup / num_threads;
                double improvement_percent = ((time_serial - time_parallel) / time_serial) * 100.0;
                double parallel_cost = num_threads * time_parallel;

                double wcss_serial = computeWCSS(points_serial, result_serial.centroids);
                double wcss_parallel = computeWCSS(points_parallel, result_parallel.centroids);

                vector<int> cluster_sizes_parallel = computeClusterSizes(points_parallel, k);

                cout << "\n===== RESULTADOS SERIAL =====\n";
                cout << "Iteraciones ejecutadas: " << result_serial.iterations << "\n";
                cout << "Tiempo serial: " << time_serial << " segundos\n";
                cout << "WCSS serial: " << fixed << setprecision(6) << wcss_serial << "\n";

                cout << "\n===== RESULTADOS PARALELOS =====\n";
                cout << "Iteraciones ejecutadas: " << result_parallel.iterations << "\n";
                cout << "Tiempo paralelo: " << time_parallel << " segundos\n";
                cout << "WCSS paralelo: " << fixed << setprecision(6) << wcss_parallel << "\n";

                cout << "\n===== METRICAS DE EFICIENCIA =====\n";
                cout << "Speedup: " << speedup << "\n";
                cout << "Eficiencia paralela: " << efficiency << "\n";
                cout << "Mejora porcentual: " << improvement_percent << "%\n";
                cout << "Costo paralelo (p * T_p): " << parallel_cost << "\n";

                printClusterSizes(cluster_sizes_parallel);
                printCentroids(result_parallel.centroids);

                writeAssignmentsCSV(output_file, points_parallel);
                cout << "\nArchivo de salida generado: " << output_file << "\n";
            }
            else if (mode == "experiment") {
                int repetitions = (argc >= 7) ? stoi(argv[6]) : 10;
                if (repetitions <= 0) {
                    throw runtime_error("El numero de repeticiones debe ser mayor que 0.");
                }

                vector<int> thread_configs = getRequiredThreadConfigs();

                cout << "Modo de ejecucion: EXPERIMENT\n";
                cout << "Repeticiones por configuracion: " << repetitions << "\n";
                cout << "Configuraciones de hilos: ";
                for (size_t i = 0; i < thread_configs.size(); ++i) {
                    cout << thread_configs[i];
                    if (i + 1 < thread_configs.size()) cout << ", ";
                }
                cout << "\n";

                writeExperimentHeaderIfNeeded(output_file);

                for (int threads : thread_configs) {
                    cout << "\n----------------------------------------\n";
                    cout << "Ejecutando experimento con " << threads << " hilo(s)\n";

                    ParallelComparisonMetrics metrics = averageComparisonMetrics(
                        original_points,
                        k,
                        threads,
                        max_iters,
                        tol,
                        repetitions,
                        base_seed
                    );

                    cout << "Promedio tiempo serial:   " << metrics.serial_time << " s\n";
                    cout << "Promedio tiempo paralelo: " << metrics.parallel_time << " s\n";
                    cout << "Promedio speedup:         " << metrics.speedup << "\n";
                    cout << "Promedio eficiencia:      " << metrics.efficiency << "\n";
                    cout << "Promedio mejora %:        " << metrics.improvement_percent << "%\n";
                    cout << "Promedio costo paralelo:  " << metrics.parallel_cost << "\n";
                    cout << "Promedio WCSS serial:     " << metrics.serial_wcss << "\n";
                    cout << "Promedio WCSS paralelo:   " << metrics.parallel_wcss << "\n";
                    cout << "Promedio iters serial:    " << metrics.serial_iterations << "\n";
                    cout << "Promedio iters paralelo:  " << metrics.parallel_iterations << "\n";

                    appendExperimentRow(
                        output_file,
                        dataset_name,
                        n_points,
                        dimension,
                        k,
                        repetitions,
                        threads,
                        metrics
                    );
                }

                cout << "\nArchivo de metricas generado: " << output_file << "\n";
                cout << "Ese CSV ya queda listo para hacer la grafica de speedup.\n";
            }
            else {
                cerr << "Error: el modo debe ser 'serial', 'parallel' o 'experiment'.\n";
                MPI_Finalize();
                return 1;
            }
        }

        MPI_Finalize();
    }
    catch (const exception& e) {
        cerr << "Error: " << e.what() << "\n";
        MPI_Finalize();
        return 1;
    }

    return 0;
}