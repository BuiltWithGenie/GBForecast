module App
using GenieFramework
using  PlotlyBase, JLD2, Statistics, DataFrames, Interpolations
using ..Utils, ..NodeUtils, ..Delhi
@genietools

# Data load and parameter set up
@load "data.jld2" train_df test_df scaling
@load "params.jld" θ
data = vcat(train_df, test_df)
features = [:meantemp, :humidity, :wind_speed, :meanpressure]
# Function to interpolate when calculating the MSE
interpolator = LinearInterpolation(data.t, data[!, :meantemp])
const N_steps = 100
_, _, init_state = neural_ode(train_df.t, length(features))
t_grid = range(minimum(data.t), maximum(data.t), length=N_steps) |> collect

# Reactive code
@app begin
     # Reactive variables
     @in r = 30
     @out e = 0.0
     @out train_data = rescale_data(train_df, features, scaling)
     @out test_data = rescale_data(test_df, features, scaling)
     @out predict_data = test_df
     # Reactive handlers
     @onchange isready,r begin
          predict_data = DataFrame( 
              t = parse_year.(rescale_t(t_grid[1:r], scaling.t_mean, scaling.t_var)),
              meantemp = predict(Vector(train_df[1, features]), t_grid[1:r], θ, 
                                  init_state, scaling.y_mean, scaling.y_var)[1, :])
          e = calc_mse(t_grid[1:r], predict_data.meantemp, interpolator, scaling)
     end
    @onchange isready begin
        for i in 30:5:100
            r = i
            sleep(0.25)
        end
    end
end
 # Route declaration
 @page("/","app.jl.html")

end
