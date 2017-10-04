function [reported_avg] = question1Knn()


A=dlmread('seeds.txt');
Acluster=A;
reported_avg = [];
%Initialize k centroidsc

k =[3 5 7];

for elm = k

    reported_avg_numerator = 0;
    
    for random_initializations = 1:10
        index = randsample(1:length(A), elm);
        centroids = A(index, :);
        sse = inf;
        for i=1:100
            %calculate distance of each point from all the centroids
            D = pdist2(A,centroids,'euclidean');

            %assign every point to a cluster and store that assignment value as the
            %8th column of the matrix - Acluster
            for j =1:size(D,1)
                [R,C] = min(D(j,:));
                Acluster(j,8) = C ;
            end

            %update all the centroid values
            for j = 1:elm
                temp_matrix = Acluster(Acluster(:,8) == j, 1:7);
                centroids(j,:) = mean(temp_matrix,1);
            end
            
            %find sse and check if the change in sse is < 0.001
            current_sse = 0;
            for j = 1:elm
                temp_matrix = Acluster(Acluster(:,8) == j, 1:7);
                dist = pdist2(temp_matrix, centroids(j,:),'euclidean');
                squared_dist = dist.^2;
                sum_squared_dist = sum(squared_dist);
                current_sse = current_sse + sum_squared_dist;
            end
            
            if sse - current_sse < 0.001
                sse = current_sse;
                break;
            else
                sse = current_sse;

            end
        end
        reported_avg_numerator = reported_avg_numerator + sse;
    end
    reported_avg = [reported_avg ; reported_avg_numerator/10];
end   
end