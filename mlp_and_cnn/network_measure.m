function res = network_measure()
    epochs = 150;
    % x = parcluster('local');
    % x.NumWorkers = 100;
    % saveProfile(x);
    % parpool(100);
    methods = {"CH3_L3"}
    datasets = {"MNIST"}
    for j = 1:2
        for i = 1:5
            for epoch = 0: 5: epochs-1
                load("./Measure/" + datasets{i} + "/" + methods{j} + "/" + epoch + ".mat")
                disp(epoch);
    
    
                tm = compute_topological_measures(adj, {"modularity", "char_path"});
                char_path = tm.char_path;
                modularity = tm.modularity;
                disp(modularity);
                save("./Measure/" + datasets{i} + "/" + methods{j} + "/" + epoch + "_measures.mat",  "modularity", "char_path");
            end
        end
    end
    end