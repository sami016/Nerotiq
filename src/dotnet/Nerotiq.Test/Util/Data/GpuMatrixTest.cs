using System;
using System.Linq;
using FluentAssertions;
using Nerotiq.Util;
using Nerotiq.Util.Data;
using OpenCL.Net;
using Xunit;

namespace Nerotiq.Test.Util.Data
{
    public class GpuMatrixTest : IClassFixture<TestScaffold>
    {
        private readonly TestScaffold _testScaffold;

        public GpuMatrixTest(TestScaffold testScaffold)
        {
            _testScaffold = testScaffold;
        }

        [Fact]
        public void Instanciate_DoesntThrow() 
        {
            var matrix = new GpuMatrix(3, 3, _testScaffold.Context);
        }
        
        [Fact]
        public void Update_DoesntThrow() 
        {
            var matrix = new GpuMatrix(3, 3, _testScaffold.Context);
            matrix.Update(
                new double[] {
                    1.0, 2.0, 3.0,
                    4.0, 5.0, 6.0,
                    7.0, 8.0, 9.0
                },
                _testScaffold.ExecutionSequence
            );
        }
        
        [Fact]
        public void Read_Works() 
        {
            var matrix = new GpuMatrix(3, 3, _testScaffold.Context);
            using (matrix.Read(_testScaffold.ExecutionSequence))
            {
                matrix.InMemoryData.Length.Should().Be(9);
                for (var i=0; i<9; i++) 
                {
                    matrix.InMemoryData[i].Should().Be(0);
                }
            }
        }
        
        [Fact]
        public void UpdateRead_Works() 
        {
            var matrix = new GpuMatrix(3, 3, _testScaffold.Context);
            matrix.Update(
                new double[] {
                    0.0, 1.0, 2.0, 
                    3.0, 4.0, 5.0, 
                    6.0, 7.0, 8.0
                },
                _testScaffold.ExecutionSequence
            );
            using (matrix.Read(_testScaffold.ExecutionSequence))
            {
                for (var i=0; i<9; i++) 
                {
                    matrix.InMemoryData[i].Should().Be(i);
                }
            }
        }

        [Fact]
        public void SetKernelArgs_Works() 
        {
            var src = @"
                __kernel void a(
                    __global double *data // 0
                ) {
                    int global_id = get_global_id(0);
                    data[global_id] *= data[global_id];
                }
            ";
            var srcs = SourceLoader.CreateProgramCollection(src);
            var program = Cl.CreateProgramWithSource(
                _testScaffold.Context.OpenClContext, 
                (uint)srcs.Length,
                srcs,
                null,
                out var error
            );
            error.Should().Be(ErrorCode.Success);
            error = Cl.BuildProgram(
                program, 
                1, 
                new[] { _testScaffold.Context.Device }, 
                string.Empty, 
                null, 
                IntPtr.Zero
            );
            error.Should().Be(ErrorCode.Success);
            var kernel = Cl.CreateKernel(program, "a", out error);
            error.Should().Be(ErrorCode.Success);

            var matrix = new GpuMatrix(3, 3, _testScaffold.Context);
            matrix.Update(
                new double[] {
                    1.0, 2.0, 3.0,
                    4.0, 5.0, 6.0,
                    7.0, 8.0, 9.0
                },
                _testScaffold.ExecutionSequence
            );
            matrix.SetKernelArg(kernel, 0);
            
            _testScaffold.ExecutionSequence.EnqueueNDRangeKernel(
                kernel,
                1,
                null,
                new IntPtr [] { new IntPtr(9) },
                null
            );

            using (matrix.Read(_testScaffold.ExecutionSequence)) 
            {
                matrix.GetValue(0, 0).Should().Be(1.0);
                matrix.GetValue(0, 1).Should().Be(4.0);
                matrix.GetValue(0, 2).Should().Be(9.0);
                matrix.GetValue(1, 0).Should().Be(16.0);
                matrix.GetValue(1, 1).Should().Be(25.0);
                matrix.GetValue(1, 2).Should().Be(36.0);
                matrix.GetValue(2, 0).Should().Be(49.0);
                matrix.GetValue(2, 1).Should().Be(64.0);
                matrix.GetValue(2, 2).Should().Be(81.0);
            }
        }

    }
}