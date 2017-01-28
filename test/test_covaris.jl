using Base.Test

IS = sparse([2. 0.1; 0.1 2.])
#IS = [2. 0; 0. 2.]

n = 2;

#A = sprandn(n,n,0.5);
#IS = A*A' + sparse_diag(n);
det(IS)

C = CovarIS(IS);
C2 = inv(full(IS));

iC = inv(C);
@test iC ≈ IS

# C times a matrix
b = randn(n,2);
a = C*b;
a2 = C2*b;
@test a ≈ a2

# C times a matrix tranposed
b = randn(n,2);
a = C*b.';
a2 = C2*b.';
@test a ≈ a2

# C times a matrix conjugate tranposed
b = randn(n,2);
a = C*b';
a2 = C2*b';
@test a ≈ a2

# C times a vector
v = randn(n);
a = C*v;
a2 = C2*v;
@test a ≈ a2

# inverse of C times a matrix
b = randn(n,2);
a = C\b;
a2 = C2\b;

@test a ≈ a2


factorize!(C);

a = C*b;
a2 = C2*b;

@test a ≈ a2


a = C\b;
a2 = C2\b;

@test a ≈ a2


@test C[1,1] ≈ C2[1,1]

@test diag(C) ≈ diag(C2)





