subroutine calc_Xika(Xika0,xi,rho,nui,nk,delta,obj_func,Xika,Xika_flat,obj_func_flat,nrho,ncomp,nbeads,nsitemax)
    implicit none
    integer,intent(in) :: nrho, ncomp, nbeads, nsitemax 
    real(8), intent(in), dimension(0:(nrho*ncomp*nbeads*nsitemax)-1) :: Xika0
    real(8), intent(in), dimension(0:ncomp-1) :: xi
    real(8), intent(in), dimension(0:ncomp-1,0:nbeads-1) :: nui
    real(8), intent(in), dimension(0:nbeads-1,0:nsitemax-1) :: nk
    real(8), intent(in), dimension(0:nrho-1, 0:ncomp-1, 0:ncomp-1, 0:nbeads-1, 0:nbeads-1, 0:nsitemax-1, 0:nsitemax-1) :: delta
    real(8), intent(in), dimension(0:nrho-1) :: rho
    real(8), intent(out), dimension(0:nrho-1,0:ncomp-1,0:nbeads-1,0:nsitemax-1) :: obj_func
    real(8), intent(out), dimension(0:nrho-1,0:ncomp-1,0:nbeads-1,0:nsitemax-1) :: Xika
    real(8), intent(out), dimension(0:(nrho*ncomp*nbeads*nsitemax)-1) :: Xika_flat,obj_func_flat
    integer :: i,k,a,j,l,b,z,r
    
    !initialize Xika to 1.0
    Xika=1.0
    z=0
    do r=0, nrho-1
    do i=0, ncomp-1
        do k=0, nbeads-1
            do a=0, nsitemax-1
                do j=0, ncomp-1
                    do l=0, nbeads-1
                        do b=0, nsitemax-1
                            Xika(r,i,k,a) = Xika(r,i,k,a) + &
                            (rho(r) * (xi(j) * nui(j,l) * nk(l,b) * Xika0(z) * delta(r,i,j,k,l,a,b)))
                            z=z+1
                        enddo
                    enddo
                enddo
            enddo
        enddo
    enddo
    enddo
    Xika = (1.0/Xika)

    
    z=0
    do r=0, nrho-1
        do i=0, ncomp-1
            do k=0, nbeads-1
                do a=0, nsitemax-1
                    Xika_flat(z)=Xika(r,i,k,a)
                    obj_func_flat(z)=Xika_flat(z)-Xika0(z)
                    z=z+1
                enddo
            enddo
        enddo
    enddo
    
end subroutine

! subroutine calc_grad_xika(Xika0,xi,rho,nui,nk,delta,step,dXika,ncomp,nbeads,nsitemax)
!     implicit none
!     integer,intent(in) :: ncomp, nbeads, nsitemax
!     real(8), intent(in) :: step
!     real(8), intent(in), dimension(0:ncomp-1, 0:nbeads-1, 0:nsitemax-1) :: Xika0
!     real(8), intent(in), dimension(0:ncomp-1) :: xi
!     real(8), intent(in), dimension(0:ncomp-1,0:nbeads-1) :: nui
!     real(8), intent(in), dimension(0:nbeads-1,0:nsitemax-1) :: nk
!     real(8), intent(in), dimension(0:ncomp-1,0:ncomp-1,0:nbeads-1,0:nbeads-1,0:nsitemax-1,0:nsitemax-1) :: delta
!     real(8), intent(in) :: rho
!     real(8), intent(out), dimension(0:(ncomp*nbeads*nsitemax)-1,0:(ncomp*nbeads*nsitemax)-1) :: dXika
!     real(8), dimension(0:ncomp-1,0:nbeads-1,0:nsitemax-1) :: obj_func_dum,Xika_p,Xika_m,Xika_dum
!     real(8), dimension(0:(ncomp*nbeads*nsitemax)-1) :: Xika_flat_p,Xika_flat_m
!     real(8), dimension(0:ncomp-1,0:nbeads-1,0:nsitemax-1) :: Xika
! 
!     integer :: i,k,a,j,l,b,z
!     
!     
!     z=0
!     do i=0, ncomp-1
!         do k=0, nbeads-1
!             do a=0, nsitemax-1
!                 Xika_p(i,k,a)=Xika0(i,k,a)+step
!                 call calc_Xika(Xika_p,xi,rho,nui,nk,delta,obj_func_dum,Xika_dum,Xika_flat_p,ncomp,nbeads,nsitemax)
!                 Xika_m(i,k,a)=Xika0(i,k,a)-step
!                 call calc_Xika(Xika_m,xi,rho,nui,nk,delta,obj_func_dum,Xika_dum,Xika_flat_m,ncomp,nbeads,nsitemax)
!                 
!                 dXika(:,z)=(Xika_flat_p-Xika_flat_m)/(2.0*step)
!             enddo
!         enddo
!     enddo
!     
! end subroutine



! subroutine min_xika(rho,Xika_init,xi,nui,nk,delta,maxiter,tol,Xika_final,nrho,ncomp,nbeads,nsitemax)
!     implicit none
!     integer, intent(in) :: nrho,ncomp,nbeads,nsitemax
!     real(8), intent(in), dimension(0:nrho-1) :: rho
!     real(8), intent(in), dimension(0:ncomp-1, 0:nbeads-1, 0:nsitemax-1) :: Xika_init
!     real(8), intent(in), dimension(0:ncomp-1) :: xi
!     real(8), intent(in), dimension(0:ncomp-1,0:nbeads-1) :: nui
!     real(8), intent(in), dimension(0:nbeads-1,0:nsitemax-1) :: nk
!     real(8), intent(in), dimension(0:nrho-1,0:ncomp-1,0:ncomp-1,0:nbeads-1,0:nbeads-1,0:nsitemax-1,0:nsitemax-1) :: delta
!     integer, intent(in) :: maxiter
!     real(8), intent(in) :: tol
!     real(8), intent(out), dimension(0:nrho-1,0:ncomp-1, 0:nbeads-1, 0:nsitemax-1) :: Xika_final
!     real(8), dimension(0:ncomp-1, 0:nbeads-1, 0:nsitemax-1) :: Xika,obj_func,Xika0
!     real(8), dimension(0:(ncomp*nbeads*nsitemax)-1) :: Xika_flat
!     integer :: i,iter
!     
!     
!     Xika=Xika_init
!     
!     do i=0, nrho-1
!         Xika0=Xika
!         !write(*,*) Xika0
!         do iter=0, maxiter-1
!             call calc_Xika(Xika0,xi,rho(i),nui,nk,delta(i,:,:,:,:,:,:),obj_func,Xika,Xika_flat,ncomp,nbeads,nsitemax)
!             call calc_grad_xika(Xika0,xi,rho,nui,nk,delta,0.000001,dXika,ncomp,nbeads,nsitemax)
!             
!             if(maxval(dabs(obj_func)).lt.tol) exit
!             Xika0=Xika
!         enddo
!         Xika_final(i,:,:,:)=Xika
!         !write(*,*) i,iter,Xika
!     end do
! end subroutine