	.section	__TEXT,__text,regular,pure_instructions
	.macosx_version_min 10, 13
	.globl	_main                   ## -- Begin function main
	.p2align	4, 0x90
_main:                                  ## @main
	.cfi_startproc
## BB#0:
	pushq	%rbp
Lcfi0:
	.cfi_def_cfa_offset 16
Lcfi1:
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
Lcfi2:
	.cfi_def_cfa_register %rbp
	andq	$-32, %rsp
	subq	$384, %rsp              ## imm = 0x180
	leaq	L_.str(%rip), %rdi
	leaq	32(%rsp), %rax
	movl	$0, 156(%rsp)
	movl	$1073741824, 220(%rsp)  ## imm = 0x40000000
	movl	$1082130432, 216(%rsp)  ## imm = 0x40800000
	movl	$1086324736, 212(%rsp)  ## imm = 0x40C00000
	movl	$1090519040, 208(%rsp)  ## imm = 0x41000000
	movl	$1092616192, 204(%rsp)  ## imm = 0x41200000
	movl	$1094713344, 200(%rsp)  ## imm = 0x41400000
	movl	$1096810496, 196(%rsp)  ## imm = 0x41600000
	movl	$1098907648, 192(%rsp)  ## imm = 0x41800000
	vmovss	212(%rsp), %xmm0        ## xmm0 = mem[0],zero,zero,zero
	vmovss	208(%rsp), %xmm1        ## xmm1 = mem[0],zero,zero,zero
	vinsertps	$16, %xmm0, %xmm1, %xmm0 ## xmm0 = xmm1[0],xmm0[0],xmm1[2,3]
	vmovss	216(%rsp), %xmm1        ## xmm1 = mem[0],zero,zero,zero
	vinsertps	$32, %xmm1, %xmm0, %xmm0 ## xmm0 = xmm0[0,1],xmm1[0],xmm0[3]
	vmovss	220(%rsp), %xmm1        ## xmm1 = mem[0],zero,zero,zero
	vinsertps	$48, %xmm1, %xmm0, %xmm0 ## xmm0 = xmm0[0,1,2],xmm1[0]
	vmovss	196(%rsp), %xmm1        ## xmm1 = mem[0],zero,zero,zero
	vmovss	192(%rsp), %xmm2        ## xmm2 = mem[0],zero,zero,zero
	vinsertps	$16, %xmm1, %xmm2, %xmm1 ## xmm1 = xmm2[0],xmm1[0],xmm2[2,3]
	vmovss	200(%rsp), %xmm2        ## xmm2 = mem[0],zero,zero,zero
	vinsertps	$32, %xmm2, %xmm1, %xmm1 ## xmm1 = xmm1[0,1],xmm2[0],xmm1[3]
	vmovss	204(%rsp), %xmm2        ## xmm2 = mem[0],zero,zero,zero
	vinsertps	$48, %xmm2, %xmm1, %xmm1 ## xmm1 = xmm1[0,1,2],xmm2[0]
                                        ## implicit-def: %YMM3
	vmovaps	%xmm1, %xmm3
	vinsertf128	$1, %xmm0, %ymm3, %ymm3
	vmovaps	%ymm3, 160(%rsp)
	vmovaps	160(%rsp), %ymm3
	vmovaps	%ymm3, 96(%rsp)
	movl	$1065353216, 364(%rsp)  ## imm = 0x3F800000
	movl	$1077936128, 360(%rsp)  ## imm = 0x40400000
	movl	$1084227584, 356(%rsp)  ## imm = 0x40A00000
	movl	$1088421888, 352(%rsp)  ## imm = 0x40E00000
	movl	$1091567616, 348(%rsp)  ## imm = 0x41100000
	movl	$1093664768, 344(%rsp)  ## imm = 0x41300000
	movl	$1095761920, 340(%rsp)  ## imm = 0x41500000
	movl	$1097859072, 336(%rsp)  ## imm = 0x41700000
	vmovss	356(%rsp), %xmm0        ## xmm0 = mem[0],zero,zero,zero
	vmovss	352(%rsp), %xmm1        ## xmm1 = mem[0],zero,zero,zero
	vinsertps	$16, %xmm0, %xmm1, %xmm0 ## xmm0 = xmm1[0],xmm0[0],xmm1[2,3]
	vmovss	360(%rsp), %xmm1        ## xmm1 = mem[0],zero,zero,zero
	vinsertps	$32, %xmm1, %xmm0, %xmm0 ## xmm0 = xmm0[0,1],xmm1[0],xmm0[3]
	vmovss	364(%rsp), %xmm1        ## xmm1 = mem[0],zero,zero,zero
	vinsertps	$48, %xmm1, %xmm0, %xmm0 ## xmm0 = xmm0[0,1,2],xmm1[0]
	vmovss	340(%rsp), %xmm1        ## xmm1 = mem[0],zero,zero,zero
	vmovss	336(%rsp), %xmm2        ## xmm2 = mem[0],zero,zero,zero
	vinsertps	$16, %xmm1, %xmm2, %xmm1 ## xmm1 = xmm2[0],xmm1[0],xmm2[2,3]
	vmovss	344(%rsp), %xmm2        ## xmm2 = mem[0],zero,zero,zero
	vinsertps	$32, %xmm2, %xmm1, %xmm1 ## xmm1 = xmm1[0,1],xmm2[0],xmm1[3]
	vmovss	348(%rsp), %xmm2        ## xmm2 = mem[0],zero,zero,zero
	vinsertps	$48, %xmm2, %xmm1, %xmm1 ## xmm1 = xmm1[0,1,2],xmm2[0]
                                        ## implicit-def: %YMM3
	vmovaps	%xmm1, %xmm3
	vinsertf128	$1, %xmm0, %ymm3, %ymm3
	vmovaps	%ymm3, 288(%rsp)
	vmovaps	288(%rsp), %ymm3
	vmovaps	%ymm3, 64(%rsp)
	vmovaps	96(%rsp), %ymm3
	vmovaps	64(%rsp), %ymm4
	vmovaps	%ymm3, 256(%rsp)
	vmovaps	%ymm4, 224(%rsp)
	vmovaps	256(%rsp), %ymm3
	vsubps	224(%rsp), %ymm3, %ymm3
	vmovaps	%ymm3, 32(%rsp)
	movq	%rax, 24(%rsp)
	movq	24(%rsp), %rax
	vmovss	(%rax), %xmm0           ## xmm0 = mem[0],zero,zero,zero
                                        ## implicit-def: %XMM1
	vcvtss2sd	%xmm0, %xmm1, %xmm0
	movq	24(%rsp), %rax
	vmovss	4(%rax), %xmm1          ## xmm1 = mem[0],zero,zero,zero
                                        ## implicit-def: %XMM2
	vcvtss2sd	%xmm1, %xmm2, %xmm1
	movq	24(%rsp), %rax
	vmovss	8(%rax), %xmm2          ## xmm2 = mem[0],zero,zero,zero
                                        ## implicit-def: %XMM5
	vcvtss2sd	%xmm2, %xmm5, %xmm2
	movq	24(%rsp), %rax
	vmovss	12(%rax), %xmm5         ## xmm5 = mem[0],zero,zero,zero
                                        ## implicit-def: %XMM6
	vcvtss2sd	%xmm5, %xmm6, %xmm3
	movq	24(%rsp), %rax
	vmovss	16(%rax), %xmm5         ## xmm5 = mem[0],zero,zero,zero
                                        ## implicit-def: %XMM6
	vcvtss2sd	%xmm5, %xmm6, %xmm4
	movq	24(%rsp), %rax
	vmovss	20(%rax), %xmm5         ## xmm5 = mem[0],zero,zero,zero
                                        ## implicit-def: %XMM6
	vcvtss2sd	%xmm5, %xmm6, %xmm5
	movq	24(%rsp), %rax
	vmovss	24(%rax), %xmm6         ## xmm6 = mem[0],zero,zero,zero
                                        ## implicit-def: %XMM7
	vcvtss2sd	%xmm6, %xmm7, %xmm6
	movq	24(%rsp), %rax
	vmovss	28(%rax), %xmm7         ## xmm7 = mem[0],zero,zero,zero
                                        ## implicit-def: %XMM8
	vcvtss2sd	%xmm7, %xmm8, %xmm7
	movb	$8, %al
	vzeroupper
	callq	_printf
	xorl	%ecx, %ecx
	movl	%eax, 20(%rsp)          ## 4-byte Spill
	movl	%ecx, %eax
	movq	%rbp, %rsp
	popq	%rbp
	retq
	.cfi_endproc
                                        ## -- End function
	.section	__TEXT,__cstring,cstring_literals
L_.str:                                 ## @.str
	.asciz	"%f %f %f %f %f %f %f %f\n"


.subsections_via_symbols
