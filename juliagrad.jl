import Base.:*
import Base.:+
import Base.:-
import Base.sum
import Base.size
using Test


TensorData = Union{Float64, Int64, Matrix}

mutable struct Param
    data::Matrix
    gradients::Vector{Tuple{Param, Function}}
    requires_grad::Bool
    grad::Union{Nothing, Param}
end

function to_matrix(a)
    if typeof(a) <: Matrix
        a
    else
        [Float64(a);;]
    end
end

function Param(data::TensorData, gradients::Vector{Tuple{Param, Function}}, requires_grad::Bool)
    m = to_matrix(data)
    Param(m, gradients, requires_grad, nothing)
end

function Param(data::TensorData)
    Param(data, Vector{Tuple{Param, Function}}(), true)
end

function Param(data::Vector)
    Param(reshape(data, :, 1))
end

function size(p::Param)
    size(p.data)
end

function sum(p::Param)
    data = sum(p.data)
    gradients = Vector{Tuple{Param, Function}}()

    if p.requires_grad
        function backward_fn(previous_grad::Param)
            Param(ones(size(p.data))) ∘ previous_grad
        end

        push!(gradients, (p, backward_fn))
    end
    Param(data, gradients, p.requires_grad)
end

function sum(p::Param, dims::Int; drop::Bool=false)
    data = sum(p.data, dims=dims)

    if drop
        data = dropdims(data, dims=dims)
    end

    gradients = Vector{Tuple{Param, Function}}()

    if p.requires_grad
        function backward_fn(previous_grad::Param)
            sum(previous_grad, dims, drop=drop)
        end

        push!(gradients, (p, backward_fn))
    end
    Param(data, gradients, p.requires_grad)
end

function +(a::Param, b::Param)
    data = a.data .+ b.data
    gradients = Vector{Tuple{Param, Function}}()
    requires_grad = a.requires_grad || b.requires_grad

    function  make_backward_fn(x::Param)
        function backward_fn(previous_grad::Param)
            result = previous_grad
            extra_dims = length(size(previous_grad)) - length(size(x))
            for i=1:extra_dims
                result = sum(result, 1, drop=true)
            end

            if size(result) == size(x)
                return result
            elseif size(x)[1] == 1 && size(result)[1] > 1
                return sum(result, 1)
            else
                error("dimensions mismatch")
            end
        end
        return backward_fn
    end

    if a.requires_grad
        push!(gradients, (a, make_backward_fn(a)))
    end

    if b.requires_grad
        push!(gradients, (b, make_backward_fn(b)))
    end
    Param(data, gradients, requires_grad)
end

function ∘(a::Param, b::Param)
    data = a.data .* b.data
    gradients = Vector{Tuple{Param, Function}}()
    requires_grad = a.requires_grad || b.requires_grad

    function  make_backward_fn(x::Param, other::Param)
        function backward_fn(previous_grad::Param)
            result = previous_grad ∘ other

            extra_dims = length(size(previous_grad)) - length(size(x))
            for i=1:extra_dims
                result = sum(result, 1, drop=true)
            end

            if size(result) == size(x)
                return result
            elseif size(x)[1] == 1 && size(result)[1] > 1
                return sum(result, 1)
            else
                error("dimensions mismatch")
            end
        end
        return backward_fn
    end

    if a.requires_grad
        push!(gradients, (a, make_backward_fn(a, b)))
    end

    if b.requires_grad
        push!(gradients, (b, make_backward_fn(b, a)))
    end
    Param(data, gradients, requires_grad)
end

function -(p::Param)
    data = -p.data
    gradients = Vector{Tuple{Param, Function}}()

    if p.requires_grad
        function backward_fn(previous_grad::Param)
            -previous_grad
        end
        push!(gradients, (p, backward_fn))
    end
    Param(data, gradients, p.requires_grad)
end

function -(a::Param, b::Param)
    a + (-b)
end

function *(a::Param, b::Param)
    data = a.data * b.data
    gradients = Vector{Tuple{Param, Function}}()
    requires_grad = a.requires_grad || b.requires_grad

    if a.requires_grad
        function  backward_fn_a(previous_grad::Param)
            previous_grad * Param(copy(b.data'))
        end
        push!(gradients, (a, backward_fn_a))
    end

    if b.requires_grad
        function backward_fn_b(previous_grad::Param)
            Param(copy(a.data')) * previous_grad
        end
        push!(gradients, (b, backward_fn_b))
    end
    Param(data, gradients, requires_grad)
end

function Base.getindex(p::Param, indeces::UnitRange...)
    # TODO Add more constructors and tests
    data = p.data[indeces...]
    gradients = Vector{Tuple{Param, Function}}()

    if p.requires_grad
        function backward_fn(previous_grad::Param)
            previous_grad[indeces...]
        end
        push!(gradients, (p, backward_fn))
    end

    Param(data, gradients, p.requires_grad)
end

function σ(p::Param)
    data = 1 ./ (1 .+ exp.(-p.data))

    gradients = Vector{Tuple{Param, Function}}()

    if p.requires_grad
        function backward_fn(previous_grad::Param)
            sigma = σ(p)
            previous_grad ∘ sigma ∘ (Param(ones(p)) - sigma)
        end
        push!(gradients, (p, backward_fn))
    end

    Param(data, gradients, p.requires_grad)
end

function tanh(p::Param)
    data = tanh(p.data)

    gradients = Vector{Tuple{Param, Function}}()

    if p.requires_grad
        function backward_fn(previous_grad::Param)
            t = tanh(p)
            previous_grad ∘ (Param(ones(p)) - (t ∘ t))
        end
        push!(gradients, (p, backward_fn))
    end

    Param(data, gradients, p.requires_grad)
end

function relu(p::Param)
    data = max.(p.data, 0)

    gradients = Vector{Tuple{Param, Function}}()

    if p.requires_grad
        function backward_fn(previous_grad::Param)
            previous_grad ∘ Param(p.data .>= 0)
        end
        push!(gradients, (p, backward_fn))
    end

    Param(data, gradients, p.requires_grad)
end


function backward(p::Param)
    @assert length(p.data) == 1
    backward(p, Param(1.0))
end

function backward(p::Param, grad::Param)
    @assert(p.requires_grad, "calling backward when requires_grad=false")

    if p.grad == nothing
        p.grad = Param(zero(p.data))
    end

    p.grad = p.grad + grad

    for (prev_param, grad_fn) in p.gradients
        grad_data = grad_fn(p.grad)
        backward(prev_param, grad_data)
    end
end




@testset "addition" begin
    p1 = Param([1 2 3])
    p2 = Param([4 5 6])
    p3 = p1 + p2

    @test p3.data == [5 7 9]

    backward(p3, Param([-1. -2. -3.]))

    @test p3.grad.data == [-1 -2 -3]
    @test p2.grad.data == [-1 -2 -3]

    p1 = Param([1 2 3; 4 5 6])
    p2 = Param([7 8 9])

    p3 = p1 + p2

    @test p3.data == [8 10 12; 11 13 15]
    backward(p3, Param([1 1 1; 1 1 1]))
    @test p1.grad.data == [1 1 1; 1 1 1]
    @test p2.grad.data == [2 2 2]
end

@testset "subtraction" begin
    p1 = Param([1 2 3])
    p2 = Param([4 5 6])

    p3 = p1 - p2
    @test p3.data == [-3 -3 -3]

    backward(p3, Param([-1. -2. -3.]))

    @test p1.grad.data == [-1 -2 -3]
    @test p2.grad.data == [1 2 3]

    p1 = Param([1 2 3; 4 5 6])
    p2 = Param([7 8 9])

    p3 = p1 - p2
    @test p3.data == [-6 -6 -6; -3 -3 -3]

    backward(p3, Param([1 1 1; 1 1 1]))

    @test p1.grad.data == [1 1 1; 1 1 1]
    @test p2.grad.data == [-2 -2 -2]
end

@testset "matrix multiplication" begin
        p1 = Param([1 2; 3 4; 5 6])
        p2 = Param([10; 20])

        p3 = p1 * p2

        @test p3.data == [50; 110; 170;;]

        grad = Param([-1; -2; -3])
        backward(p3, grad)

        @test p1.grad.data == grad.data * p2.data'
        @test p2.grad.data == p1.data' * grad.data
end

@testset "sum to scalar" begin
    p1 = Param([1 2 3])
    p2 = sum(p1)
    backward(p2)

    @test p1.grad.data == [1 1 1]

    p1 = Param([1 2 3])
    p2 = sum(p1)

    backward(p2, Param(3.0))
    @test p1.grad.data == [3 3 3]
end;

@testset "hadamard product" begin
    p1 = Param([1 2 3])
    p2 = Param([4 5 6])

    p3 = p1 ∘ p2

    @test p3.data == [4 10 18]

    backward(p3, Param([-1. -2. -3.]))

    @test p1.grad.data == [-4 -10 -18]
    @test p2.grad.data == [-1  -4  -9]

    p1 = Param([1 2 3; 4 5 6])
    p2 = Param([7 8 9])

    p3 = p1 ∘ p2

    @test p3.data == [7 16 27; 28 40 54]

    backward(p3, Param([1 1 1; 1 1 1]))

    @test p1.grad.data == [7 8 9; 7 8 9]
    @test p2.grad.data == [5 7 9]
end
