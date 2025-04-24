package ptithcm.entity;

import java.util.Date;

import javax.persistence.Column;
import javax.persistence.Entity;
import javax.persistence.GeneratedValue;
import javax.persistence.Id;
import javax.persistence.JoinColumn;
import javax.persistence.ManyToOne;
import javax.persistence.Table;
import javax.persistence.Temporal;
import javax.persistence.TemporalType;

import org.springframework.format.annotation.DateTimeFormat;

@Entity
@Table(name="DANHGIA")
public class DanhGiaEntity {
	@Id
	@Column(name="MADG")
	@GeneratedValue
	private Integer maDg;
	@Column(name = "SOSAO")
	private float soSao;
	@Column(name = "NOIDUNG") 
	private String noiDung;
	@Temporal(TemporalType.DATE)
	@DateTimeFormat(pattern = "yyyy-mm-dd")
	@Column(name = "NGAY") 
	private Date ngay; 
	@ManyToOne
	@JoinColumn(name = "MAND") 
	private NguoiDungEntity nguoiDung; 
	@ManyToOne
	@JoinColumn(name = "MASP") 
	private SanPhamEntity sanPham;
	public Integer getMaDg() {
		return maDg;
	}
	public void setMaDg(Integer maDg) {
		this.maDg = maDg;
	}
	public float getSoSao() {
		return soSao;
	}
	
	public void setSoSao(float soSao) {
		this.soSao = soSao;
	}
	public String getNoiDung() {
		return noiDung;
	}
	public void setNoiDung(String noiDung) {
		this.noiDung = noiDung;
	}
	public Date getNgay() {
		return ngay;
	}
	public void setNgay(Date ngay) {
		this.ngay = ngay;
	}
	public NguoiDungEntity getNguoiDung() {
		return nguoiDung;
	}
	public void setNguoiDung(NguoiDungEntity nguoiDung) {
		this.nguoiDung = nguoiDung;
	}
	public SanPhamEntity getSanPham() {
		return sanPham;
	}
	public void setSanPham(SanPhamEntity sanPham) {
		this.sanPham = sanPham;
	}
	
	
}


 
	 	  