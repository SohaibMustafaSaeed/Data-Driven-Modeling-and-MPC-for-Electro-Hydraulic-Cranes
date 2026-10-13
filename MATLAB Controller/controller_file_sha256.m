function result = controller_file_sha256(filename)
% Core MATLAB SHA-256; no Java or toolbox dependency.
fid = fopen(filename, 'rb');
if fid < 0
    error('CraneController:HashFile', 'Cannot read %s.', filename);
end
cleanup = onCleanup(@() fclose(fid));
data = fread(fid, Inf, '*uint8');
bits = uint64(numel(data)) * uint64(8);
padding = mod(56 - mod(numel(data) + 1, 64), 64);
data = [data; uint8(128); zeros(padding, 1, 'uint8')];
for j = 7:-1:0
    data(end + 1, 1) = uint8(bitand(bitshift(bits, -8 * j), uint64(255)));
end
k = uint32(hex2dec({ ...
    '428a2f98';'71374491';'b5c0fbcf';'e9b5dba5';'3956c25b';'59f111f1';'923f82a4';'ab1c5ed5'; ...
    'd807aa98';'12835b01';'243185be';'550c7dc3';'72be5d74';'80deb1fe';'9bdc06a7';'c19bf174'; ...
    'e49b69c1';'efbe4786';'0fc19dc6';'240ca1cc';'2de92c6f';'4a7484aa';'5cb0a9dc';'76f988da'; ...
    '983e5152';'a831c66d';'b00327c8';'bf597fc7';'c6e00bf3';'d5a79147';'06ca6351';'14292967'; ...
    '27b70a85';'2e1b2138';'4d2c6dfc';'53380d13';'650a7354';'766a0abb';'81c2c92e';'92722c85'; ...
    'a2bfe8a1';'a81a664b';'c24b8b70';'c76c51a3';'d192e819';'d6990624';'f40e3585';'106aa070'; ...
    '19a4c116';'1e376c08';'2748774c';'34b0bcb5';'391c0cb3';'4ed8aa4a';'5b9cca4f';'682e6ff3'; ...
    '748f82ee';'78a5636f';'84c87814';'8cc70208';'90befffa';'a4506ceb';'bef9a3f7';'c67178f2'}));
hash = uint32(hex2dec({'6a09e667';'bb67ae85';'3c6ef372';'a54ff53a';'510e527f';'9b05688c';'1f83d9ab';'5be0cd19'}));
for offset = 1:64:numel(data)
    w = zeros(64, 1, 'uint32');
    for j = 1:16
        p = offset + (j - 1) * 4;
        w(j) = bitor(bitor(bitshift(uint32(data(p)), 24), bitshift(uint32(data(p + 1)), 16)), ...
            bitor(bitshift(uint32(data(p + 2)), 8), uint32(data(p + 3))));
    end
    for j = 17:64
        s0 = bitxor(bitxor(rotate(w(j - 15), 7), rotate(w(j - 15), 18)), bitshift(w(j - 15), -3));
        s1 = bitxor(bitxor(rotate(w(j - 2), 17), rotate(w(j - 2), 19)), bitshift(w(j - 2), -10));
        w(j) = add32(w(j - 16), s0, w(j - 7), s1);
    end
    a = hash(1); b = hash(2); c = hash(3); d = hash(4);
    e = hash(5); f = hash(6); g = hash(7); h = hash(8);
    for j = 1:64
        s1 = bitxor(bitxor(rotate(e, 6), rotate(e, 11)), rotate(e, 25));
        choose = bitxor(bitand(e, f), bitand(bitcmp(e), g));
        t1 = add32(h, s1, choose, k(j), w(j));
        s0 = bitxor(bitxor(rotate(a, 2), rotate(a, 13)), rotate(a, 22));
        majority = bitxor(bitxor(bitand(a, b), bitand(a, c)), bitand(b, c));
        t2 = add32(s0, majority);
        h = g; g = f; f = e; e = add32(d, t1);
        d = c; c = b; b = a; a = add32(t1, t2);
    end
    values = [a;b;c;d;e;f;g;h];
    for j = 1:8
        hash(j) = add32(hash(j), values(j));
    end
end
result = lower(reshape(dec2hex(hash, 8).', 1, []));
end

function y = rotate(x, n)
y = bitor(bitshift(x, -n), bitshift(x, 32 - n));
end

function y = add32(varargin)
total = uint64(0);
for j = 1:nargin
    total = total + uint64(varargin{j});
end
y = uint32(bitand(total, uint64(4294967295)));
end
